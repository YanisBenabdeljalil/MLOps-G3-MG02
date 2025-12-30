terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-west-3" # Région Paris
}

# --- 1. INFRASTRUCTURE DE STOCKAGE ---

resource "aws_s3_bucket" "data_bucket" {
  bucket        = "s3-g3mg02" 
  force_destroy = true
}

resource "aws_ecr_repository" "ml_repo" {
  name                 = "ecr-g3mg02" 
  force_delete         = true
  image_tag_mutability = "MUTABLE"
}

# --- 2. GESTION DES RÔLES (IAM) ---

# Création du rôle qui permet à App Runner de récupérer les images sur ECR
resource "aws_iam_role" "apprunner_service_role" {
  name = "apprunner-service-role-g3mg02"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
      }
    ]
  })
}

# Attacher la permission ECR au rôle
resource "aws_iam_role_policy_attachment" "apprunner_ecr_access" {
  role       = aws_iam_role.apprunner_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# --- 3. SERVICE BACKEND (API FastAPI) ---

resource "aws_apprunner_service" "api_service" {
  service_name = "apprunner-backend-g3mg02" 

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_service_role.arn
    }
    image_repository {
      image_identifier      = "${aws_ecr_repository.ml_repo.repository_url}:latest"
      image_repository_type = "ECR"
      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          "ENV" = "PROD"
          "S3_BUCKET" = aws_s3_bucket.data_bucket.id
        }
      }
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu    = "1024"
    memory = "2048"
  }
}

# --- 4. SERVICE FRONTEND (Streamlit) ---

resource "aws_apprunner_service" "frontend_service" {
  service_name = "apprunner-frontend-g3mg02"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_service_role.arn
    }
    
    image_repository {
      # On pointe vers le tag spécifique :frontend
      image_identifier      = "${aws_ecr_repository.ml_repo.repository_url}:frontend"
      image_repository_type = "ECR"
      
      image_configuration {
        port = "8501" # Port Streamlit
        
        runtime_environment_variables = {
          # Injection automatique de l'URL du Backend créé au-dessus
          API_URL = "https://${aws_apprunner_service.api_service.service_url}"
        }
      }
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu    = "1024"
    memory = "2048"
  }
}

# --- 5. OUTPUTS (Liens finaux) ---

output "s3_bucket_name" {
  value = aws_s3_bucket.data_bucket.id
}

output "ecr_repository_url" {
  value = aws_ecr_repository.ml_repo.repository_url
}

output "backend_api_url" {
  description = "Lien vers l'API (Swagger dispo sur /docs)"
  value       = "https://${aws_apprunner_service.api_service.service_url}"
}

output "frontend_app_url" {
  description = "Lien final vers ton application Streamlit"
  value       = "https://${aws_apprunner_service.frontend_service.service_url}"
}