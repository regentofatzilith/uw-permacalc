# GitHub Repository Setup Guide

## Step 1: Create Repository on GitHub

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in details:
   - **Repository name**: `uw-permacalc`
   - **Description**: "Ultimate Weapon Permanence Calculator for The Tower"
   - **Visibility**: Public (required for free Render deployment)
   - **Initialize**: Do NOT check any boxes (we have files already)
4. Click **"Create repository"**

## Step 2: Push Your Code

Open PowerShell in the `deploy_render` folder and run these commands:

```powershell
# Navigate to the deploy folder
cd "\\mycloudex2ultra\Thorsten\_python\ProjectAtzi\deploy_render"

# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: UW PermaCalc v1.0"

# Add your GitHub repository as remote
# Replace YOUR-USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR-USERNAME/uw-permacalc.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### If you get authentication errors:

**Option A: Use GitHub Personal Access Token**
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. When prompted for password, use the token instead

**Option B: Use GitHub Desktop**
1. Download GitHub Desktop: https://desktop.github.com/
2. File → Add Local Repository → Select `deploy_render` folder
3. Click "Publish repository"

## Step 3: Verify Upload

Go to your GitHub repo URL:
```
https://github.com/YOUR-USERNAME/uw-permacalc
```

You should see:
- ✅ perma_calc_new.py
- ✅ requirements.txt
- ✅ README.md
- ✅ .gitignore

## Step 4: Deploy to Render

Now that your code is on GitHub:

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect account"** to link GitHub (if not already linked)
4. Find your `uw-permacalc` repository and click **"Connect"**
5. Configure:
   - **Name**: `uw-permacalc`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn perma_calc_new:server`
   - **Instance Type**: Free
6. Click **"Create Web Service"**

Render will build and deploy (2-5 minutes). You'll get a URL like:
```
https://uw-permacalc.onrender.com
```

## Step 5: Update README

After deployment, update the README.md with your actual URL:

```powershell
# Edit README.md and replace "your-app-name.onrender.com"
# Then commit and push:
git add README.md
git commit -m "Update live demo URL"
git push
```

Render will auto-deploy the change.

## Troubleshooting

### "Permission denied" when pushing
- Check you're logged into the correct GitHub account
- Use Personal Access Token instead of password
- Or use GitHub Desktop

### "Repository not found"
- Double-check the repository URL
- Make sure you created the repo on GitHub first
- Verify you have write access

### Files not showing on GitHub
- Check you're in the right folder: `deploy_render`
- Make sure `git add .` was run before commit
- Try `git status` to see what's staged

## Quick Commands Reference

```powershell
# Check status
git status

# Add new files
git add .

# Commit changes
git commit -m "Your message here"

# Push to GitHub
git push

# Pull latest from GitHub
git pull
```

## Next Steps

Once deployed:
1. ✅ Test the live URL
2. ✅ Share the link
3. ✅ Monitor Render dashboard for logs
4. ✅ Make updates by pushing to GitHub (auto-deploys)

---

Your app will be live and accessible to anyone with the URL - no installation required! 🚀
