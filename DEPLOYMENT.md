# 🚀 Cyberpunk Churn Prediction - Deployment Guide

## 🎮 Deployment Options

This guide covers multiple deployment options for the Cyberpunk Churn Prediction dashboard.

---

## 📋 Prerequisites

- Git installed
- GitHub account
- Python 3.11+ (for local development)
- Docker (for containerized deployment)

---

## 🌐 **Option 1: Streamlit Cloud (Recommended)**

### Steps:
1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy cyberpunk churn prediction app"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io/)
   - Connect your GitHub account
   - Select your repository: `churn-assignment`
   - Set main file: `app.py`
   - Click "Deploy"

3. **Configuration:**
   - The `.streamlit/config.toml` file is already configured with cyberpunk theme
   - No additional setup required

### ✅ **Pros:**
- Free hosting
- Automatic deployments on git push
- Built-in SSL
- Easy to use

---

## 🐳 **Option 2: Docker Deployment**

### Local Docker:
```bash
# Build the image
docker build -t cyberpunk-churn-prediction .

# Run the container
docker run -d -p 8501:8501 --name cyberpunk-churn cyberpunk-churn-prediction
```

### Docker Compose:
```bash
# Start the application
docker-compose up -d

# Stop the application
docker-compose down
```

### ✅ **Pros:**
- Consistent environment
- Easy scaling
- Production-ready

---

## 🚂 **Option 3: Railway**

### Steps:
1. **Push to GitHub** (same as above)

2. **Deploy to Railway:**
   - Visit [railway.app](https://railway.app/)
   - Connect GitHub account
   - Select your repository
   - Railway will automatically detect the `railway.json` configuration

3. **Configuration:**
   - The `railway.json` file is pre-configured
   - Uses Dockerfile for deployment

### ✅ **Pros:**
- Simple deployment
- Automatic scaling
- Built-in monitoring

---

## 🟣 **Option 4: Heroku**

### Steps:
1. **Install Heroku CLI**

2. **Create Heroku App:**
   ```bash
   heroku create your-cyberpunk-churn-app
   ```

3. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### ✅ **Pros:**
- Easy deployment
- Add-ons available
- Good documentation

---

## ☁️ **Option 5: Cloud Platforms**

### AWS (using ECS/Fargate):
1. Push Docker image to ECR
2. Create ECS task definition
3. Deploy to Fargate

### Google Cloud (using Cloud Run):
1. Push to Container Registry
2. Deploy to Cloud Run
3. Configure custom domain

### Azure (using Container Instances):
1. Push to Azure Container Registry
2. Deploy to Container Instances
3. Configure load balancer

---

## 🔧 **Environment Variables**

For production deployments, set these environment variables:

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 📁 **File Structure for Deployment**

```
churn-assignment/
├── streamlit_app_simple.py      # Main application
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose setup
├── Procfile                     # Heroku configuration
├── runtime.txt                  # Python version for Heroku
├── railway.json                 # Railway configuration
├── setup.sh                     # Heroku setup script
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── .dockerignore               # Docker ignore file
└── DEPLOYMENT.md               # This file
```

---

## 🎯 **Quick Deploy Commands**

### Streamlit Cloud:
```bash
git add . && git commit -m "Deploy app" && git push origin main
```

### Docker:
```bash
docker-compose up -d
```

### Railway:
```bash
# Just push to GitHub, Railway auto-deploys
git push origin main
```

### Heroku:
```bash
git push heroku main
```

---

## 🔍 **Troubleshooting**

### Common Issues:

1. **Port Issues:**
   - Ensure the app uses `$PORT` environment variable
   - Default port is 8501 for local, dynamic for cloud

2. **Memory Issues:**
   - Optimize requirements.txt
   - Use lighter base images

3. **Build Failures:**
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt

4. **Theme Not Loading:**
   - Ensure `.streamlit/config.toml` is included
   - Check CSS is properly embedded

---

## 🎮 **Post-Deployment**

After successful deployment:

1. **Test the Application:**
   - Verify cyberpunk theme loads correctly
   - Test all navigation pages
   - Check responsive design on mobile

2. **Monitor Performance:**
   - Check loading times
   - Monitor resource usage
   - Set up health checks

3. **Custom Domain (Optional):**
   - Configure custom domain
   - Set up SSL certificate
   - Update DNS records

---

## 🎉 **Success!**

Your Cyberpunk Churn Prediction dashboard should now be live with:

- ✅ **Dark cyberpunk theme** with neon accents
- ✅ **Responsive design** for all devices
- ✅ **Interactive dashboard** with multiple pages
- ✅ **Professional deployment** ready for production

**🎮 Enjoy your cyberpunk ML dashboard!**
