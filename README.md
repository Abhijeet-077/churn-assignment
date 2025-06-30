# 🎮 Cyberpunk Churn Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-green)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A stunning cyberpunk-themed Streamlit dashboard for customer churn prediction with neon aesthetics and interactive visualizations.

## ✨ Features

- 🌑 **Dark Cyberpunk Theme** - Neon colors with dark backgrounds
- ⚡ **Interactive Dashboard** - Multiple pages with smooth navigation
- 📊 **Data Visualizations** - Cyberpunk-styled charts and graphs
- 🎯 **Churn Prediction** - AI-powered customer churn simulation
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile
- 🎨 **Neon Aesthetics** - Cyan, purple, green, and pink accents
- 🔤 **Cyberpunk Fonts** - Orbitron and Rajdhani typography

## 🚀 Quick Start

### Option 1: Streamlit Cloud (Recommended)

1. Visit [share.streamlit.io](https://share.streamlit.io/)
2. Connect your GitHub account
3. Select this repository
4. Set main file: `app.py`
5. Click "Deploy!"

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/Abhijeet-077/churn-assignment.git
cd churn-assignment

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 3: Docker

```bash
# Build and run with Docker
docker build -t cyberpunk-churn .
docker run -p 8501:8501 cyberpunk-churn
```

## 📱 Dashboard Pages

### 🏠 Dashboard
- Overview metrics with neon cards
- Customer statistics
- Interactive charts

### 🎯 Prediction
- Customer data input form
- Real-time churn prediction
- Risk level assessment

### 📊 Data Explorer
- Sample customer data
- Interactive visualizations
- Data analysis tools

### 🎮 Theme Demo
- Color palette showcase
- Interactive elements
- Cyberpunk styling examples

## 🎨 Cyberpunk Theme

The dashboard features a complete cyberpunk aesthetic:

- **Colors**: Neon cyan (#00ffff), purple (#ff00ff), green (#00ff00), pink (#ff0040)
- **Backgrounds**: Dark themes (#0d1117, #1a1a1a, #2d2d2d)
- **Typography**: Orbitron (headers) and Rajdhani (body)
- **Effects**: Glowing text, neon borders, smooth animations

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Styling**: Custom CSS with cyberpunk theme
- **Deployment**: Docker, Streamlit Cloud, Heroku, Railway

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.0.0
```

## 🚀 Deployment Options

### Streamlit Cloud
- **Free hosting**
- **Automatic deployments**
- **Built-in SSL**

### Railway
- **Easy deployment**
- **Automatic scaling**
- **Built-in monitoring**

### Heroku
- **Simple deployment**
- **Add-ons available**
- **Good documentation**

### Docker
- **Consistent environment**
- **Easy scaling**
- **Production-ready**

## 📁 Project Structure

```
churn-assignment/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── Procfile                 # Heroku configuration
├── railway.json             # Railway configuration
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # This file
```

## 🎯 Features Showcase

### Interactive Prediction
- Customer data input forms
- Real-time churn probability calculation
- Risk level visualization
- Confidence metrics

### Data Visualization
- Churn distribution charts
- Customer segmentation analysis
- Interactive scatter plots
- Cyberpunk-styled graphs

### Responsive Design
- Mobile-friendly interface
- Tablet optimization
- Desktop experience
- Touch-friendly controls

## 🔧 Configuration

The app uses Streamlit's configuration system:

```toml
[theme]
base = "dark"
primaryColor = "#00ffff"
backgroundColor = "#0d1117"
secondaryBackgroundColor = "#1a1a1a"
textColor = "#ffffff"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the application
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎮 Live Demo

Visit the live demo: [Cyberpunk Churn Prediction](https://your-app-name.streamlit.app)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the deployment documentation
- Review the troubleshooting guide

---

**🎮 Built with cyberpunk aesthetics for the future of data science dashboards!**
