###############################################################################
# Financial-IA — GCP Infrastructure (Terraform)
#
# Resources:
#   1. GCS Datalake Bucket (Standard, us-central1)
#   2. Data Ingest VM (e2-standard-4, preemptible)
#   3. GPU Training VM (A100 Spot, Deep Learning VM image)
#   4. Service Account with minimal permissions
#   5. Firewall rules for SSH
#
# Usage:
#   cd infra/
#   terraform init
#   terraform plan -var="project_id=YOUR_PROJECT"
#   terraform apply -var="project_id=YOUR_PROJECT"
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ---------------------------------------------------------------------------
# Service Account (least-privilege)
# ---------------------------------------------------------------------------

resource "google_service_account" "financial_ia" {
  account_id   = "financial-ia-sa"
  display_name = "Financial IA Service Account"
}

resource "google_project_iam_member" "sa_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.financial_ia.email}"
}

resource "google_project_iam_member" "sa_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.financial_ia.email}"
}

resource "google_project_iam_member" "sa_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.financial_ia.email}"
}

# ---------------------------------------------------------------------------
# 1. GCS Datalake Bucket
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "datalake" {
  name          = var.bucket_name
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = false
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 90 # Move to Nearline after 90 days
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age = 365 # Move to Coldline after 1 year
    }
  }
}

# ---------------------------------------------------------------------------
# 2. Data Ingest VM (e2-standard-4, preemptible for cost savings)
# ---------------------------------------------------------------------------

resource "google_compute_instance" "ingest" {
  name         = "financial-ia-ingest"
  machine_type = var.ingest_machine_type
  zone         = var.zone

  scheduling {
    preemptible       = true
    automatic_restart = false
  }

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2404-lts-amd64"
      size  = 50 # GB — minimal, data goes to GCS
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {} # Ephemeral public IP
  }

  service_account {
    email  = google_service_account.financial_ia.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(var.ssh_pub_key_path)}"
  }

  metadata_startup_script = file("${path.module}/startup_ingest.sh")

  tags = ["financial-ia", "ingest", "ssh"]

  labels = {
    project = "financial-ia"
    role    = "ingest"
  }
}

# ---------------------------------------------------------------------------
# 3. GPU Training VM (A100 Spot, Deep Learning VM)
# ---------------------------------------------------------------------------

resource "google_compute_instance" "training" {
  name         = "financial-ia-training"
  machine_type = var.training_machine_type
  zone         = var.zone

  scheduling {
    provisioning_model = "SPOT"
    preemptible        = true
    automatic_restart  = false
    # Spot VMs: instance_termination_action defaults to STOP
  }

  boot_disk {
    initialize_params {
      # Google Deep Learning VM with PyTorch + CUDA pre-installed
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu"
      size  = 200 # GB for datasets + checkpoints
      type  = "pd-ssd"
    }
  }

  # A2 machine types have GPUs built-in (no guest_accelerator needed)
  # If using N1 + attached GPU, uncomment below:
  # guest_accelerator {
  #   type  = var.training_gpu_type
  #   count = 1
  # }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    email  = google_service_account.financial_ia.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    ssh-keys                = "${var.ssh_user}:${file(var.ssh_pub_key_path)}"
    install-nvidia-driver   = "True"
    proxy-mode              = "project_editors"
  }

  metadata_startup_script = file("${path.module}/startup_training.sh")

  tags = ["financial-ia", "training", "ssh"]

  labels = {
    project = "financial-ia"
    role    = "training"
  }
}

# ---------------------------------------------------------------------------
# Firewall: allow SSH
# ---------------------------------------------------------------------------

resource "google_compute_firewall" "ssh" {
  name    = "financial-ia-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh"]
}

# Firewall: allow TensorBoard (optional, port 6006)
resource "google_compute_firewall" "tensorboard" {
  name    = "financial-ia-allow-tensorboard"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["6006"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["training"]
}
