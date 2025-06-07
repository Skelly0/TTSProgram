# GitHub Repository Setup Guide

## Steps to Create Private Repository and Push Code

Since GitHub CLI is not available, here are the manual steps to create a private repository on GitHub:

### Step 1: Create Repository on GitHub Website

1. Go to [GitHub.com](https://github.com) and sign in to your account
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - **Repository name**: `document-to-audiobook-converter` (or your preferred name)
   - **Description**: `Convert text documents to audiobooks using Kokoro-82M TTS model`
   - **Visibility**: Select "Private" ✓
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/document-to-audiobook-converter.git

# Rename the default branch to main (if needed)
git branch -M main

# Push the code to GitHub
git push -u origin main
```

### Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all the project files uploaded
3. The repository should be marked as "Private"

## Alternative: Using GitHub CLI (if you want to install it)

If you prefer to use GitHub CLI for future projects:

### Install GitHub CLI
- **Windows**: Download from https://cli.github.com/ or use `winget install GitHub.cli`
- **macOS**: `brew install gh`
- **Linux**: Follow instructions at https://github.com/cli/cli#installation

### Commands with GitHub CLI
```bash
# Login to GitHub
gh auth login

# Create private repository and push
gh repo create document-to-audiobook-converter --private --source=. --remote=origin --push
```

## Repository Structure

Your private repository will contain:
```
document-to-audiobook-converter/
├── .gitignore                 # Git ignore rules
├── document_to_audiobook.py   # Main program
├── requirements.txt           # Python dependencies
├── setup.py                  # Setup script
├── run_converter.bat         # Windows runner
├── run_converter.sh          # Unix/Linux/Mac runner
├── README.md                 # Documentation
├── PROJECT_OVERVIEW.md       # Quick start guide
├── GITHUB_SETUP.md          # This file
└── documents/               # Sample documents
    └── sample_story.txt     # Example file
```

## Next Steps After Upload

1. **Clone on other machines**: `git clone https://github.com/YOUR_USERNAME/document-to-audiobook-converter.git`
2. **Make changes**: Edit files, then `git add .`, `git commit -m "message"`, `git push`
3. **Collaborate**: Add collaborators in repository settings if needed
4. **Backup**: Your code is now safely backed up on GitHub

## Security Note

Since this is a private repository:
- Only you (and collaborators you add) can see the code
- The repository won't appear in search results
- You can change it to public later if desired in repository settings
