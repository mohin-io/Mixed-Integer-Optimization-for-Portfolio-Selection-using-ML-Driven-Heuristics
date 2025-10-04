# Streamlit Cloud Deployment Guide
## Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics

**Estimated Time:** 5-10 minutes
**Difficulty:** Beginner
**Cost:** FREE

---

## 🎯 Prerequisites

Before deploying, ensure you have:
- ✅ GitHub account (free)
- ✅ Repository pushed to GitHub (already done ✓)
- ✅ All deployment files ready (already configured ✓)

Your repository is already configured with:
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `Procfile` - Deployment configuration
- ✅ `setup.sh` - Setup script
- ✅ `requirements.txt` - Python dependencies
- ✅ `src/visualization/dashboard.py` - Main app

---

## 📝 Step-by-Step Deployment Instructions

### Step 1: Access Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Open your browser
   - Navigate to: **https://share.streamlit.io**

2. **Sign In:**
   - Click **"Sign in"** button (top right)
   - Choose **"Continue with GitHub"**
   - Authorize Streamlit to access your GitHub account
   - Grant necessary permissions when prompted

---

### Step 2: Create New App

1. **Click "New app" button** (on the dashboard)

2. **Fill in deployment settings:**

   ```
   Repository:  mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection
   Branch:      master
   Main file:   src/visualization/dashboard.py
   ```

   **Detailed Fields:**
   - **Repository:** Select your repository from the dropdown
     - If not listed, click "Paste GitHub URL" and enter:
       `https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection`

   - **Branch:** `master` (or `main` if that's your default branch)

   - **Main file path:** `src/visualization/dashboard.py`

3. **Advanced Settings (Optional):**
   - Click "Advanced settings" to customize:
     - **App URL:** Choose custom subdomain (e.g., `portfolio-optimizer.streamlit.app`)
     - **Python version:** 3.10 or 3.11 (auto-detected from `runtime.txt`)
     - **Environment variables:** None needed for this app

---

### Step 3: Deploy!

1. **Click the "Deploy!" button**

2. **Wait for deployment** (2-3 minutes):
   - You'll see a build log showing:
     - ✓ Cloning repository
     - ✓ Installing dependencies
     - ✓ Building app
     - ✓ Starting server

3. **Deployment Complete:**
   - Your app URL will be displayed
   - Format: `https://[your-app-name].streamlit.app`

---

## 🎉 Your App is Live!

### Access Your Deployed App

**Your app will be available at:**
```
https://[your-app-name].streamlit.app
```

**Example:**
```
https://portfolio-optimizer.streamlit.app
```

---

## 🔧 Post-Deployment Configuration

### Share Your App

1. **Get Shareable Link:**
   - Copy your app URL
   - Share with anyone (no login required for viewers)

2. **Embed in Portfolio/Resume:**
   ```markdown
   🔗 [Live Demo](https://your-app-name.streamlit.app)
   ```

3. **Add to GitHub README:**
   ```markdown
   ## 🚀 Live Demo
   Try the live application: [Portfolio Optimizer](https://your-app-name.streamlit.app)
   ```

### Manage Your App

**From Streamlit Cloud Dashboard:**
- 📊 **View Analytics** - Usage stats, visitors
- 🔄 **Reboot App** - Restart if needed
- ⚙️ **Settings** - Update configuration
- 🗑️ **Delete App** - Remove deployment

---

## 🔄 Automatic Updates

**Your app auto-updates when you push to GitHub!**

Every time you push to your `master` branch:
1. Streamlit Cloud detects the change
2. Automatically rebuilds your app
3. Deploys the new version
4. **No manual redeployment needed!**

To update your app:
```bash
# Make changes locally
git add .
git commit -m "Your update message"
git push origin master

# Streamlit Cloud will auto-update in 2-3 minutes
```

---

## 📱 Features of Your Deployed App

Your deployed dashboard includes:

### Interactive Features:
- 📊 **Portfolio Optimization** - 4 strategies
  - Equal Weight
  - Max Sharpe
  - Min Variance
  - Concentrated

- 📈 **Real-time Visualization** - 4 tabs
  - Portfolio Weights (bar chart + table)
  - Asset Prices (time series)
  - Correlation Matrix (heatmap)
  - Performance (cumulative returns)

- ⚙️ **Dynamic Controls** - Sidebar
  - Number of assets (5-20)
  - Historical days (250-2000)
  - Random seed (1-1000)
  - Strategy-specific parameters

### Performance Metrics:
- Expected Annual Return
- Annual Volatility
- Sharpe Ratio
- Number of Active Assets

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Issue 1: "Module not found" error
**Solution:**
- Check `requirements.txt` includes all dependencies
- Update requirements:
  ```bash
  pip freeze > requirements.txt
  git add requirements.txt
  git commit -m "fix: update requirements"
  git push
  ```

#### Issue 2: App crashes on startup
**Solution:**
- Check Streamlit Cloud logs (in deployment dashboard)
- Verify `src/visualization/dashboard.py` path is correct
- Ensure Python version compatibility

#### Issue 3: Slow loading
**Solution:**
- This is normal for first load (cold start)
- Subsequent loads are faster
- Consider optimizing data loading

#### Issue 4: "Port already in use"
**Solution:**
- This shouldn't happen on Streamlit Cloud
- If testing locally, use: `streamlit run src/visualization/dashboard.py --server.port 8502`

---

## 🔒 Security & Privacy

### What Streamlit Cloud Provides:
- ✅ **HTTPS/SSL** - Automatic secure connection
- ✅ **DDoS Protection** - Built-in protection
- ✅ **Uptime Monitoring** - 99.9% availability
- ✅ **CDN** - Fast global access

### What You Should Know:
- 🔓 App is **publicly accessible** by default
- 🔐 For private apps, upgrade to Streamlit Cloud Teams
- 📊 No sensitive data is stored (uses synthetic/demo data)
- 🔑 No authentication required for viewers

---

## 📊 Monitoring Your App

### View App Analytics

1. **Go to Streamlit Cloud dashboard**
2. **Click on your app**
3. **View metrics:**
   - 👥 Number of viewers
   - 📈 Usage over time
   - 🌍 Geographic distribution
   - ⏱️ Session duration

### Check Logs

**To view logs:**
1. Click "Manage app" in Streamlit Cloud
2. Click "Logs" tab
3. See real-time application logs
4. Debug any issues

---

## 🚀 Advanced: Custom Domain (Optional)

### Use Your Own Domain

**Requirements:**
- Streamlit Cloud Teams plan (paid)
- Custom domain ownership

**Steps:**
1. Upgrade to Teams plan
2. Go to app settings
3. Add custom domain
4. Update DNS records
5. Verify domain

**Example:**
```
portfolio.yourdomain.com
```

---

## 📈 Performance Optimization Tips

### Speed Up Your App:

1. **Cache Data Loading:**
   ```python
   @st.cache_data
   def load_data():
       # Your data loading code
   ```

2. **Cache Computations:**
   ```python
   @st.cache_resource
   def expensive_computation():
       # Your computation
   ```

3. **Optimize Asset Loading:**
   - Reduce default number of assets
   - Use smaller date ranges for demos
   - Lazy load visualizations

4. **Reduce Cold Starts:**
   - Streamlit Cloud free tier may sleep after inactivity
   - First load after sleep takes 30-60 seconds
   - Upgrade to paid tier for always-on apps

---

## 🎯 Quick Reference

### Essential Commands

**Local Testing:**
```bash
streamlit run src/visualization/dashboard.py
```

**Update Deployed App:**
```bash
git add .
git commit -m "update: description"
git push origin master
```

**Check Deployment Status:**
- Visit: https://share.streamlit.io/[username]/[repo]/[branch]/[file]

---

## 📞 Support & Resources

### If You Need Help:

1. **Streamlit Documentation:**
   - https://docs.streamlit.io/

2. **Streamlit Community Forum:**
   - https://discuss.streamlit.io/

3. **GitHub Issues:**
   - https://github.com/streamlit/streamlit/issues

4. **Your Project Issues:**
   - https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection/issues

---

## ✅ Deployment Checklist

Use this checklist to verify your deployment:

**Pre-Deployment:**
- [x] GitHub repository is public
- [x] All code pushed to GitHub
- [x] `requirements.txt` is up to date
- [x] `Procfile` is configured
- [x] `.streamlit/config.toml` exists
- [x] Main file path is correct

**During Deployment:**
- [ ] Signed in to Streamlit Cloud
- [ ] Created new app
- [ ] Selected correct repository
- [ ] Selected correct branch (master)
- [ ] Entered correct main file path
- [ ] Clicked "Deploy!"

**Post-Deployment:**
- [ ] App deployed successfully
- [ ] App loads without errors
- [ ] All features work correctly
- [ ] Shared app URL
- [ ] Added to README/portfolio

---

## 🎊 Congratulations!

Your **Mixed-Integer-Optimization-for-Portfolio-Selection-using-ML-Driven-Heuristics** app is now live on Streamlit Cloud!

### Share Your Achievement:

**Add to your resume/portfolio:**
```
Portfolio Optimization Dashboard | Python, Streamlit, ML
• Deployed production-ready web application to Streamlit Cloud
• Features 4 optimization strategies with real-time visualization
• Live demo: https://your-app-name.streamlit.app
```

**Share on LinkedIn:**
```
🚀 Excited to share my latest project!

I've deployed a sophisticated Portfolio Optimization Dashboard
using Mixed-Integer Programming and ML-driven heuristics.

✅ 4 optimization strategies
✅ Real-time interactive visualizations
✅ Comprehensive backtesting framework

Live demo: https://your-app-name.streamlit.app
GitHub: https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection

#Python #MachineLearning #Finance #DataScience #Optimization
```

---

## 🔗 Quick Links

- **Streamlit Cloud:** https://share.streamlit.io
- **Your Repository:** https://github.com/mohin-io/Mixed-Integer-Optimization-for-Portfolio-Selection
- **Streamlit Docs:** https://docs.streamlit.io
- **Deployment Guide:** https://docs.streamlit.io/streamlit-community-cloud/get-started

---

**Deployment Status:** ✅ Ready to Deploy
**Estimated Time:** 5 minutes
**Difficulty:** ⭐ Easy
**Cost:** FREE

**Go deploy your app now at: https://share.streamlit.io** 🚀
