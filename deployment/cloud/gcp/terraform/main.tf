# Terraform configuration for GeoAnomalyMapper (GAM) on Google Cloud Platform
# This deploys scalable compute, storage, and serverless resources for GAM processing.
# Prerequisites: gcloud auth login, terraform init, terraform apply -var="project_id=your-project"

terraform {
  required_version = ">= 1.5.0"
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

# Variables (use terraform.tfvars or -var for overrides)
variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "gam-project"  # Replace with actual
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be 'dev' or 'prod'."
  }
}

variable "machine_type" {
  description = "Machine type for GAM instances"
  type        = string
  default     = "e2-medium"
}

variable "min_instances" {
  description = "Minimum instances in MIG"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum instances in MIG"
  type        = number
  default     = 5
}

variable "input_bucket_name" {
  description = "Name for input data bucket"
  type        = string
  default     = "gam-input-data"
}

variable "output_bucket_name" {
  description = "Name for output data bucket"
  type        = string
  default     = "gam-output-data"
}

# Networking
resource "google_compute_network" "gam_vpc" {
  name                    = "${var.environment}-gam-vpc"
  auto_create_subnetworks = false
  description             = "VPC for GAM deployment"
}

resource "google_compute_subnetwork" "public_subnet" {
  name          = "${var.environment}-public-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.gam_vpc.id
  description   = "Public subnet for GAM compute"
}

resource "google_compute_subnetwork" "private_subnet" {
  name          = "${var.environment}-private-subnet"
  ip_cidr_range = "10.0.2.0/24"
  region        = var.region
  network       = google_compute_network.gam_vpc.id
  description   = "Private subnet for storage access"
}

resource "google_compute_router" "gam_router" {
  name    = "${var.environment}-gam-router"
  region  = var.region
  network = google_compute_network.gam_vpc.id
}

resource "google_compute_address" "gam_static_ip" {
  name    = "${var.environment}-gam-ip"
  region  = var.region
  purpose = "GCE_INSTANCE"
}

# Firewall (allow SSH, HTTP for health checks)
resource "google_compute_firewall" "gam_firewall" {
  name    = "${var.environment}-gam-firewall"
  network = google_compute_network.gam_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "8000", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]  # Restrict to bastion in prod
  target_tags   = ["gam-instance"]
}

# IAM Service Account
resource "google_service_account" "gam_sa" {
  account_id   = "${var.environment}-gam-sa"
  display_name = "Service account for GAM instances"
}

resource "google_project_iam_member" "gam_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.gam_sa.email}"
}

resource "google_project_iam_member" "gam_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gam_sa.email}"
}

resource "google_project_iam_member" "gam_monitoring_viewer" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.gam_sa.email}"
}

# Compute: Instance Template
resource "google_compute_instance_template" "gam_template" {
  name_prefix  = "${var.environment}-gam-template-"
  machine_type = var.machine_type
  region       = var.region

  tags = ["gam-instance"]

  disk {
    boot         = true
    source_image = "projects/debian-cloud/global/images/family/debian-12"
    auto_delete  = true
    disk_size_gb = 30
    type         = "pd-standard"
  }

  network_interface {
    network    = google_compute_network.gam_vpc.id
    subnetwork = google_compute_subnetwork.public_subnet.id
    access_config {
      nat_ip = google_compute_address.gam_static_ip.address
    }
  }

  service_account {
    email  = google_service_account.gam_sa.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      apt-get update
      apt-get install -y docker.io
      systemctl start docker
      systemctl enable docker
      # Pull GAM image from Artifact Registry (assume pushed)
      gcloud auth configure-docker ${var.region}-docker.pkg.dev
      docker pull ${var.region}-docker.pkg.dev/${var.project_id}/gam-repo/gam:latest
      # Run GAM (example: monitor S3-like bucket events or periodic run)
      docker run --rm \
        -v /tmp:/app/data \
        -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
        ${var.region}-docker.pkg.dev/${var.project_id}/gam-repo/gam:latest \
        gam run --config /app/config/production.yaml --input-gcs ${google_storage_bucket.input_data.name} --output-gcs ${google_storage_bucket.output_data.name}
      EOF
  }

  scheduling {
    preemptible       = var.environment == "dev" ? true : false
    automatic_restart = true
  }
}

# Managed Instance Group
resource "google_compute_region_instance_group_manager" "gam_mig" {
  name               = "${var.environment}-gam-mig"
  region             = var.region
  instance_template  = google_compute_instance_template.gam_template.id
  base_instance_name = "${var.environment}-gam-instance"

  version {
    name = "initial"
  }

  named_port {
    name = "http"
    port = 8000
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.gam_health.id
    initial_delay_sec = 60
  }

  update_policy {
    type                           = "OPPORTUNISTIC"
    minimal_action                 = "REPLACE"
    most_disruptive_allowed_action = "REPLACE"
    replacement_method             = "RECREATE"
  }
}

# Health Check
resource "google_compute_health_check" "gam_health" {
  name               = "${var.environment}-gam-health"
  check_interval_sec = 30
  timeout_sec        = 10
  healthy_threshold  = 2
  unhealthy_threshold = 3

  tcp_health_check {
    port = 8000
  }
}

# Autoscaler
resource "google_compute_region_autoscaler" "gam_autoscaler" {
  name   = "${var.environment}-gam-autoscaler"
  region = var.region
  target = google_compute_region_instance_group_manager.gam_mig.id

  autoscaling_policy {
    max_replicas    = var.max_instances
    min_replicas    = var.min_instances
    cooldown_period = 60

    cpu_utilization {
      target = 0.6
    }
  }
}

# Storage Buckets
resource "google_storage_bucket" "input_data" {
  name                        = "${var.input_bucket_name}-${var.environment}-${var.project_id}"
  location                    = var.region
  force_destroy               = var.environment == "dev" ? true : false
  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }
  encryption {
    default_kms_key_name = null  # Use Google-managed
  }
  labels = {
    environment = var.environment
    type        = "input"
  }
}

resource "google_storage_bucket" "output_data" {
  name                        = "${var.output_bucket_name}-${var.environment}-${var.project_id}"
  location                    = var.region
  force_destroy               = var.environment == "dev" ? true : false
  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }
  encryption {
    default_kms_key_name = null
  }
  labels = {
    environment = var.environment
    type        = "output"
  }
}

# Cloud Function for Event-Driven Processing (trigger on input bucket upload)
resource "google_storage_bucket_iam_member" "function_invoker" {
  bucket = google_storage_bucket.input_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gam_sa.email}"
}

data "archive_file" "gam_function" {
  type        = "zip"
  output_path = "/tmp/gam_function.zip"
  source {
    content = <<-EOF
      import functions_framework
      from google.cloud import storage
      import subprocess
      import os

      @functions_framework.cloud_event
      def trigger_gam_analysis(cloud_event):
          data = cloud_event.data
          bucket = data["bucket"]
          name = data["name"]
          print(f"Processing file {name} from bucket {bucket}")
          # Run GAM CLI in container (simplified; use Cloud Run for complex)
          cmd = [
              "docker", "run", "--rm",
              "-v", f"/tmp:/app/data",
              "-e", "GOOGLE_CLOUD_PROJECT=${var.project_id}",
              f"gcr.io/{var.project_id}/gam:latest",
              "gam", "run", "--input-gcs", bucket, "--file", name
          ]
          result = subprocess.run(cmd, capture_output=True)
          if result.returncode != 0:
              raise Exception(f"GAM run failed: {result.stderr}")
          print("GAM analysis completed")
      EOF
    filename = "main.py"
  }
}

resource "google_cloudfunctions2_function" "gam_trigger" {
  name        = "${var.environment}-gam-trigger"
  location    = var.region
  description = "Triggers GAM analysis on new data upload"

  build_config {
    runtime     = "python312"
    entry_point = "trigger_gam_analysis"
    source_archives {
      src = data.archive_file.gam_function.output_path
    }
  }

  event_trigger {
    event_type = "google.cloud.storage.object.v1.finalized"
    bucket     = google_storage_bucket.input_data.name
  }

  service_config {
    available_memory   = "256M"
    timeout_seconds    = 540  # 9 min for GAM processing
    service_account_email = google_service_account.gam_sa.email
  }
}

# Outputs
output "input_bucket_name" {
  description = "Name of input data bucket"
  value       = google_storage_bucket.input_data.name
}

output "output_bucket_name" {
  description = "Name of output data bucket"
  value       = google_storage_bucket.output_data.name
}

output "mig_self_link" {
  description = "Self-link of the Managed Instance Group"
  value       = google_compute_region_instance_group_manager.gam_mig.self_link
}

output "function_uri" {
  description = "URI of the Cloud Function"
  value       = google_cloudfunctions2_function.gam_trigger.uri
}

output "vpc_id" {
  description = "VPC ID"
  value       = google_compute_network.gam_vpc.id
}