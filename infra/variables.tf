###############################################################################
# Variables — Financial-IA GCP Infrastructure
###############################################################################

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for VMs"
  type        = string
  default     = "us-central1-a"
}

variable "bucket_name" {
  description = "GCS datalake bucket name (must be globally unique)"
  type        = string
  default     = "financial-ia-datalake"
}

variable "ingest_machine_type" {
  description = "Machine type for data ingestion VM"
  type        = string
  default     = "e2-standard-4"
}

variable "training_machine_type" {
  description = "Machine type for GPU training VM"
  type        = string
  default     = "a2-highgpu-1g" # 1x A100 40GB
}

variable "training_gpu_type" {
  description = "GPU type for training VM (only for N1/custom VMs)"
  type        = string
  default     = "nvidia-tesla-a100"
}

variable "ssh_user" {
  description = "SSH username for VM access"
  type        = string
  default     = "daniel"
}

variable "ssh_pub_key_path" {
  description = "Path to SSH public key"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}
