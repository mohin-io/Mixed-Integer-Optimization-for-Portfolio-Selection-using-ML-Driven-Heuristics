# Step-by-Step Deployment Guide
## Streamlit Portfolio Optimization Dashboard

**Status:** ✅ Ready for Deployment
**Last Updated:** 2025-10-04

---

## 🎯 Pre-Deployment Checklist

Before deploying, verify all prerequisites are met:

- [x] All tests passing (63/63) ✅
- [x] Code pushed to GitHub ✅
- [x] `requirements.txt` present ✅
- [x] `.streamlit/config.toml` configured ✅
- [x] Documentation complete ✅
- [x] No secrets or API keys in code ✅

**Repository:** https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection

---

## 🚀 Option 1: Streamlit Cloud (Recommended)

**Best for:** Quick deployment, free hosting, automatic updates

### Step 1: Prepare Your Account

1. **Visit Streamlit Cloud**
   ```
   https://share.streamlit.io
   ```

2. **Sign in with GitHub**
   - Click "Sign in with GitHub"
   - Authorize Streamlit to access your repositories
   - Grant necessary permissions

### Step 2: Deploy the App

1. **Click "New app"** in the Streamlit Cloud dashboard

2. **Configure the deployment:**
   ```
   Repository: mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection
   Branch: master
   Main file path: src/visualization/dashboard.py
   ```

3. **Click "Deploy!"**

### Step 3: Wait for Deployment

The deployment process takes 2-5 minutes:

```
[1/4] Cloning repository...          ✅
[2/4] Installing dependencies...     ✅
[3/4] Building app...               ✅
[4/4] Starting server...            ✅
```

### Step 4: Verify Deployment

Once deployed, you'll get a URL like:
```
https://mohin-io-mixed-integer-optimization.streamlit.app
```

**Test the following:**
- [x] App loads without errors
- [x] Sidebar controls work
- [x] All 4 strategies run successfully
- [x] Visualizations render correctly
- [x] No console errors

### Step 5: Share Your App

Your app is now live! Share the URL:
- Add to your resume/portfolio
- Share on social media
- Include in project README
- Send to recruiters

---

## 🐳 Option 2: Docker Deployment

**Best for:** Local testing, custom hosting, reproducibility

### Step 1: Build Docker Image

```bash
cd Mixed-Integer-Optimization-for-Portfolio-Selection

# Build the image
docker build -t portfolio-optimizer .

# Verify build
docker images | grep portfolio-optimizer
```

### Step 2: Run Container

```bash
# Run on port 8501
docker run -p 8501:8501 portfolio-optimizer

# Or use docker-compose
docker-compose up
```

### Step 3: Access the App

Open your browser:
```
http://localhost:8501
```

### Step 4: Stop Container

```bash
# Find container ID
docker ps

# Stop container
docker stop <container_id>

# Or with docker-compose
docker-compose down
```

---

## ☁️ Option 3: Heroku Deployment

**Best for:** Production hosting, custom domain, scalability

### Step 1: Install Heroku CLI

```bash
# Download from:
https://devcenter.heroku.com/articles/heroku-cli

# Verify installation
heroku --version
```

### Step 2: Login to Heroku

```bash
heroku login
# Opens browser for authentication
```

### Step 3: Create Heroku App

```bash
cd Mixed-Integer-Optimization-for-Portfolio-Selection

# Create app
heroku create portfolio-optimizer-unique-name

# Verify
heroku apps:info
```

### Step 4: Deploy to Heroku

```bash
# Add Heroku remote (if not auto-added)
heroku git:remote -a portfolio-optimizer-unique-name

# Push to Heroku
git push heroku master

# Or from different branch
git push heroku main:master
```

### Step 5: Verify Deployment

```bash
# Open app in browser
heroku open

# View logs
heroku logs --tail

# Check status
heroku ps
```

### Step 6: Scale (Optional)

```bash
# Scale up
heroku ps:scale web=1

# Check dyno status
heroku ps
```

---

## 🖥️ Option 4: AWS EC2 Deployment

**Best for:** Full control, enterprise deployment, custom configuration

### Step 1: Launch EC2 Instance

1. **Login to AWS Console**
   - Navigate to EC2
   - Click "Launch Instance"

2. **Configure Instance:**
   ```
   AMI: Ubuntu Server 22.04 LTS
   Instance Type: t2.micro (free tier) or t2.small
   Storage: 20 GB
   Security Group: Allow ports 22 (SSH) and 8501 (Streamlit)
   ```

3. **Create/Select Key Pair**
   - Download `.pem` file
   - Save securely

### Step 2: Connect to Instance

```bash
# Set permissions
chmod 400 your-key.pem

# Connect via SSH
ssh -i your-key.pem ubuntu@<your-ec2-ip>
```

### Step 3: Setup Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip git -y

# Clone repository
git clone https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection.git

# Navigate to directory
cd Mixed-Integer-Optimization-for-Portfolio-Selection
```

### Step 4: Install Dependencies

```bash
# Install requirements
pip3 install -r requirements.txt

# Verify installation
python3 -c "import streamlit; print(streamlit.__version__)"
```

### Step 5: Run Dashboard

```bash
# Run Streamlit (background process)
nohup streamlit run src/visualization/dashboard.py --server.port 8501 --server.address 0.0.0.0 &

# Check if running
ps aux | grep streamlit
```

### Step 6: Access Dashboard

```
http://<your-ec2-ip>:8501
```

### Step 7: Setup as Service (Optional)

Create systemd service for auto-restart:

```bash
sudo nano /etc/systemd/system/portfolio-dashboard.service
```

Add:
```ini
[Unit]
Description=Portfolio Optimization Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mixed-Integer-Optimization-for-Portfolio-Selection
ExecStart=/usr/local/bin/streamlit run src/visualization/dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable portfolio-dashboard
sudo systemctl start portfolio-dashboard
sudo systemctl status portfolio-dashboard
```

---

## 🔒 Security Considerations

### For All Deployments

1. **No Secrets in Code**
   - ✅ No API keys
   - ✅ No passwords
   - ✅ No sensitive data

2. **HTTPS Enabled**
   - Streamlit Cloud: Automatic ✅
   - Heroku: Automatic ✅
   - EC2: Configure with Let's Encrypt

3. **Environment Variables**
   - Use for configuration
   - Never commit `.env` files
   - Use platform-specific secret management

### Streamlit Cloud Specific

- Secrets managed in dashboard settings
- Automatic HTTPS
- DDoS protection included

### Heroku Specific

```bash
# Set environment variables
heroku config:set KEY=VALUE

# View config
heroku config
```

### AWS EC2 Specific

- Configure Security Groups properly
- Use IAM roles, not credentials
- Enable AWS CloudWatch for monitoring
- Consider AWS Application Load Balancer

---

## 📊 Monitoring & Maintenance

### Streamlit Cloud

**View Logs:**
- Click on app in dashboard
- Navigate to "Logs" tab
- Real-time streaming logs

**Restart App:**
- Click "Reboot app" in dashboard
- Automatic restart on code push

**Metrics:**
- View in Streamlit Cloud dashboard
- Active users, requests, errors

### Heroku

**View Logs:**
```bash
heroku logs --tail
heroku logs --num 100
```

**Metrics:**
```bash
heroku ps
heroku status
```

**Restart:**
```bash
heroku restart
```

### AWS EC2

**View Logs:**
```bash
# Streamlit logs
tail -f nohup.out

# System logs
sudo journalctl -u portfolio-dashboard -f
```

**Monitoring:**
- Use AWS CloudWatch
- Set up alarms for CPU/memory
- Monitor disk space

**Restart:**
```bash
sudo systemctl restart portfolio-dashboard
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Or specific package
pip install streamlit --upgrade
```

#### Issue: "Port already in use"

**Solution:**
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run src/visualization/dashboard.py --server.port 8502
```

#### Issue: "App not loading"

**Solution:**
1. Check logs for errors
2. Verify all dependencies installed
3. Check firewall/security groups
4. Try clearing browser cache
5. Test locally first

#### Issue: "Optimization too slow"

**Solution:**
- Reduce number of assets
- Use smaller time window
- Check server resources
- Consider upgrading instance type

### Platform-Specific Issues

**Streamlit Cloud:**
- Check repository permissions
- Verify branch name (master vs main)
- Check file paths (case-sensitive)
- Review deployment logs

**Heroku:**
- Check Procfile syntax
- Verify buildpack
- Check dyno status
- Review app logs

**AWS EC2:**
- Verify security groups
- Check instance status
- Monitor resource usage
- Review systemd logs

---

## 🔄 Updating the App

### Streamlit Cloud

Automatic updates on Git push:
```bash
# Make changes locally
git add .
git commit -m "Update dashboard"
git push origin master

# Streamlit Cloud auto-deploys
```

### Heroku

Manual deployment:
```bash
# Push changes
git push heroku master

# Verify deployment
heroku logs --tail
```

### AWS EC2

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<ec2-ip>

# Pull latest changes
cd Mixed-Integer-Optimization-for-Portfolio-Selection
git pull origin master

# Restart service
sudo systemctl restart portfolio-dashboard
```

---

## ✅ Post-Deployment Checklist

After deploying, verify:

- [ ] App loads without errors
- [ ] All sidebar controls work
- [ ] Data generation successful (all parameter combinations)
- [ ] All 4 strategies optimize correctly
  - [ ] Equal Weight
  - [ ] Max Sharpe
  - [ ] Min Variance
  - [ ] Concentrated
- [ ] All visualizations render
  - [ ] Portfolio Weights
  - [ ] Asset Prices
  - [ ] Correlation Matrix
  - [ ] Performance Chart
- [ ] Metrics display correctly
- [ ] No console errors
- [ ] Acceptable performance (<30s optimizations)
- [ ] Mobile responsive (test on phone)
- [ ] Share URL works publicly

---

## 📈 Success Metrics

Monitor these metrics post-deployment:

**Technical:**
- Uptime: Target 99.9%
- Response time: < 3s initial load
- Optimization time: < 30s
- Error rate: < 0.1%

**User Engagement:**
- Daily active users
- Average session duration
- Feature usage (which strategies most popular)
- Bounce rate

**Performance:**
- Memory usage
- CPU usage
- Request latency
- Concurrent users

---

## 🎉 Deployment Complete!

Once deployed successfully, your dashboard is ready for:

✅ **Portfolio in resume/CV**
✅ **Demonstration to recruiters**
✅ **Research and education**
✅ **Further development**
✅ **Community contributions**

---

## 📞 Support

**Issues with deployment?**
- Open GitHub Issue: https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection/issues
- Email: mohinhasin999@gmail.com
- Check documentation in `docs/` folder

---

## 📚 Related Documentation

- [USER_GUIDE.md](USER_GUIDE.md) - How to use the dashboard
- [DEPLOYMENT.md](DEPLOYMENT.md) - Detailed deployment options
- [TEST_REPORT.md](../TEST_REPORT.md) - Testing results
- [README.md](../README.md) - Project overview

---

**Version:** 1.0.0
**Last Updated:** 2025-10-04
**Status:** Production Ready 🚀
