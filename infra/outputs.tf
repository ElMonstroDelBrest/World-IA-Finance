###############################################################################
# Outputs — Financial-IA GCP Infrastructure
###############################################################################

output "bucket_url" {
  description = "GCS datalake bucket URL"
  value       = "gs://${google_storage_bucket.datalake.name}"
}

output "ingest_vm_ip" {
  description = "External IP of the data ingest VM"
  value       = google_compute_instance.ingest.network_interface[0].access_config[0].nat_ip
}

output "training_vm_ip" {
  description = "External IP of the GPU training VM"
  value       = google_compute_instance.training.network_interface[0].access_config[0].nat_ip
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.financial_ia.email
}

output "ssh_ingest" {
  description = "SSH command for ingest VM"
  value       = "gcloud compute ssh ${google_compute_instance.ingest.name} --zone=${var.zone}"
}

output "ssh_training" {
  description = "SSH command for training VM"
  value       = "gcloud compute ssh ${google_compute_instance.training.name} --zone=${var.zone}"
}
