# Deployment Guide

This guide covers deploying the Portfolio Optimization system to various platforms.

---

## 📋 Table of Contents

1. [Streamlit Cloud Deployment](#streamlit-cloud)
2. [Heroku Deployment](#heroku)
3. [Docker Deployment](#docker)
4. [AWS Deployment](#aws)
5. [Local Deployment](#local)

---

## 🎈 Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [streamlit.io](https://streamlit.io/cloud))

### Steps

1. **Fork/Clone Repository**
   ```bash
   git clone https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git
   ```

2. **Sign in to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub

3. **Deploy App**
   - Click "New app"
   - Repository: `mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection`
   - Branch: `master`
   - Main file path: `src/visualization/dashboard.py`
   - Click "Deploy!"

4. **Configure (Optional)**
   - Add secrets in Streamlit Cloud dashboard
   - Set environment variables if needed

5. **Access Your App**
   - URL will be: `https://[your-app-name].streamlit.app`

### Configuration Files
- `.streamlit/config.toml` - Theme and server settings
- `runtime.txt` - Python version specification

---

## 🚀 Heroku Deployment

### Prerequisites
- Heroku account ([signup.heroku.com](https://signup.heroku.com))
- Heroku CLI installed

### Steps

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku

   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli

   # Linux
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   cd Mixed-Integer-Optimization-for-Portfolio-Selection
   heroku create portfolio-optimizer-app
   ```

4. **Deploy**
   ```bash
   git push heroku master
   ```

5. **Scale Dyno**
   ```bash
   heroku ps:scale web=1
   ```

6. **Open App**
   ```bash
   heroku open
   ```

### Required Files
- `Procfile` - Defines app startup
- `setup.sh` - Streamlit configuration script
- `runtime.txt` - Python version
- `requirements.txt` - Dependencies

### Troubleshooting
```bash
# View logs
heroku logs --tail

# Restart app
heroku restart

# Check dyno status
heroku ps
```

---

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t portfolio-optimizer .

# Run container
docker run -p 8501:8501 portfolio-optimizer

# Access at http://localhost:8501
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Deploy to Docker Hub

```bash
# Login
docker login

# Tag image
docker tag portfolio-optimizer mohin-io/portfolio-optimizer:latest

# Push
docker push mohin-io/portfolio-optimizer:latest
```

### Deploy to Cloud with Docker

#### AWS ECS
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [account-id].dkr.ecr.us-east-1.amazonaws.com

docker tag portfolio-optimizer:latest [account-id].dkr.ecr.us-east-1.amazonaws.com/portfolio-optimizer:latest

docker push [account-id].dkr.ecr.us-east-1.amazonaws.com/portfolio-optimizer:latest
```

#### Google Cloud Run
```bash
# Tag for GCR
docker tag portfolio-optimizer gcr.io/[project-id]/portfolio-optimizer

# Push
docker push gcr.io/[project-id]/portfolio-optimizer

# Deploy
gcloud run deploy portfolio-optimizer \
  --image gcr.io/[project-id]/portfolio-optimizer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## ☁️ AWS Deployment

### EC2 Deployment

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t2.medium or larger
   - Security group: Allow ports 22 (SSH), 8501 (Streamlit)

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@ec2-instance-ip

   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python 3.10
   sudo apt install python3.10 python3.10-venv python3-pip -y

   # Clone repository
   git clone https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git
   cd Mixed-Integer-Optimization-for-Portfolio-Selection

   # Setup virtual environment
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run with systemd**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/portfolio-optimizer.service
   ```

   Add:
   ```ini
   [Unit]
   Description=Portfolio Optimizer Dashboard
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/Mixed-Integer-Optimization-for-Portfolio-Selection
   Environment="PATH=/home/ubuntu/Mixed-Integer-Optimization-for-Portfolio-Selection/venv/bin"
   ExecStart=/home/ubuntu/Mixed-Integer-Optimization-for-Portfolio-Selection/venv/bin/streamlit run src/visualization/dashboard.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   # Enable and start
   sudo systemctl enable portfolio-optimizer
   sudo systemctl start portfolio-optimizer
   sudo systemctl status portfolio-optimizer
   ```

4. **Setup Nginx (Optional)**
   ```bash
   sudo apt install nginx -y

   # Configure reverse proxy
   sudo nano /etc/nginx/sites-available/portfolio-optimizer
   ```

   Add:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

   ```bash
   sudo ln -s /etc/nginx/sites-available/portfolio-optimizer /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

---

## 💻 Local Deployment

### Development Server

```bash
# Clone repository
git clone https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git
cd Mixed-Integer-Optimization-for-Portfolio-Selection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run src/visualization/dashboard.py
```

### Production-like Local Setup

```bash
# Using gunicorn + nginx (for FastAPI in future)
pip install gunicorn

# Or use Docker
docker-compose up
```

---

## 🔒 Security Considerations

### Environment Variables
Never commit sensitive data. Use environment variables:

```bash
# .env file (add to .gitignore)
API_KEY=your-api-key
DATABASE_URL=postgresql://user:password@host:port/db
SECRET_KEY=your-secret-key
```

Load in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
```

### HTTPS/SSL
- **Streamlit Cloud**: Automatic HTTPS
- **Heroku**: Automatic HTTPS
- **EC2**: Use Let's Encrypt
  ```bash
  sudo apt install certbot python3-certbot-nginx
  sudo certbot --nginx -d your-domain.com
  ```

### Firewall
```bash
# UFW on Ubuntu
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## 📊 Monitoring & Logging

### Streamlit Cloud
- Built-in logs in dashboard
- Usage analytics

### Heroku
```bash
# View logs
heroku logs --tail --app portfolio-optimizer-app

# Papertrail addon
heroku addons:create papertrail:choklad
```

### Docker
```bash
# View container logs
docker logs -f [container-id]

# Use logging drivers
docker run --log-driver=syslog portfolio-optimizer
```

### AWS CloudWatch
```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb
```

---

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process
lsof -i :8501

# Kill process
kill -9 [PID]
```

**Memory Issues**
```bash
# Increase swap (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Dependency Conflicts**
```bash
# Clear pip cache
pip cache purge

# Reinstall
pip install --force-reinstall -r requirements.txt
```

---

## 📈 Performance Optimization

### Streamlit Caching
Already implemented with `@st.cache_data` decorators

### Database Connection Pooling
For future SQL integration:
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://...',
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10
)
```

### CDN for Static Assets
Use Cloudflare or AWS CloudFront for serving visualizations

---

## 🔄 Automatic Deployment

**Streamlit Cloud**: Auto-deploys on push to master

---

## 📞 Support

For deployment issues:
- **GitHub Issues**: https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection/issues
- **Email**: mohinhasin999@gmail.com

---

**Last Updated**: October 2025
**Status**: ✅ Production-Ready
