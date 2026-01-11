# GitHub Setup Without Git Command Line

## Option 1: GitHub Desktop (Recommended - Easiest)

### Step 1: Install GitHub Desktop

1. Download: https://desktop.github.com/
2. Install the application
3. Sign in with your GitHub account

### Step 2: Create Repository on GitHub.com

1. Go to https://github.com
2. Click **"+"** (top right) → **"New repository"**
3. Name: `uw-permacalc`
4. Description: "Ultimate Weapon Permanence Calculator for The Tower"
5. **Public** (required for free Render)
6. ✅ **Check "Add a README file"** (this time we DO check it)
7. Click **"Create repository"**

### Step 3: Clone to Your Computer

1. Open GitHub Desktop
2. File → Clone Repository
3. Find `uw-permacalc` in the list
4. Choose location: `C:\Users\YourUsername\Documents\GitHub\uw-permacalc`
5. Click **Clone**

### Step 4: Copy Your Files

1. Open File Explorer
2. Navigate to: `\\mycloudex2ultra\Thorsten\_python\ProjectAtzi\deploy_render`
3. **Copy** these files:
   - `perma_calc_new.py`
   - `requirements.txt`
   - `.gitignore`
4. Navigate to where you cloned the repo (e.g., `C:\Users\YourUsername\Documents\GitHub\uw-permacalc`)
5. **Paste** the files there
6. **Replace** the existing README.md with the one from deploy_render folder

### Step 5: Commit and Push

1. GitHub Desktop will show all changed files
2. Bottom left: Enter commit message: `Initial commit: UW PermaCalc v1.0`
3. Click **"Commit to main"**
4. Click **"Push origin"** (top right)

✅ **Done!** Your code is now on GitHub!

---

## Option 2: Direct Web Upload (No Installation)

### Step 1: Create Repository

Same as above - create `uw-permacalc` repo on GitHub.com

### Step 2: Upload Files via Web

1. On your repository page, click **"Add file"** → **"Upload files"**
2. Drag and drop from: `\\mycloudex2ultra\Thorsten\_python\ProjectAtzi\deploy_render`
   - `perma_calc_new.py`
   - `requirements.txt`
   - `.gitignore`
3. **Commit message**: `Add PermaCalc files`
4. Click **"Commit changes"**

### Step 3: Update README

1. Click on `README.md` in your repo
2. Click the pencil icon (edit)
3. Delete the default content
4. Open `deploy_render\README.md` on your computer
5. Copy all content
6. Paste into GitHub editor
7. Click **"Commit changes"**

✅ **Done!**

---

## Option 3: Install Git (For Future Use)

Download Git for Windows: https://git-scm.com/download/win

Then you can use the git commands from the original guide.

---

## After Upload - Deploy to Render

Once files are on GitHub (using any method above):

1. Go to https://dashboard.render.com
2. **New +** → **Web Service**
3. **Connect GitHub** account
4. Select `uw-permacalc` repository
5. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn perma_calc_new:server`
   - **Free** tier
6. **Create Web Service**

Your app will be live in 3-5 minutes! 🚀

---

## I Recommend: GitHub Desktop

It's the easiest option and gives you a nice GUI for future updates. Just download, install, and follow the steps above.
