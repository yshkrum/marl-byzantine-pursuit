#!/usr/bin/env bash
# Run this once after creating the repo on GitHub.
# Replace <org> with your GitHub username or org name.
# Replace <repo> with marl-byzantine-pursuit (or your chosen name).

set -e

ORG="<org>"
REPO="marl-byzantine-pursuit"

echo "==> Initialising git"
git init
git add .
git commit -m "init: project scaffold, folder structure, stubs and configs"

echo "==> Adding remote"
git remote add origin "https://github.com/$ORG/$REPO.git"

echo "==> Pushing main"
git branch -M main
git push -u origin main

echo "==> Creating dev branch"
git checkout -b dev
git push -u origin dev

echo ""
echo "Done. Branch structure:"
echo "  main  — stable, reviewed code only"
echo "  dev   — integration branch, all PRs target here"
echo ""
echo "Each team member now runs:"
echo "  git checkout dev && git pull origin dev"
echo "  git checkout -b <prefix>/<TICKET-ID>-description"
