"""
Deployment verification script for Cyberpunk Churn Prediction app.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_deployment_readiness():
    """Check if all deployment files are ready."""
    
    print("🎮 CYBERPUNK CHURN PREDICTION - DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    # Essential files
    essential_files = [
        ("streamlit_app_simple.py", "Main Streamlit application"),
        ("requirements.txt", "Python dependencies"),
        ("README.md", "Project documentation"),
    ]
    
    # Deployment configuration files
    deployment_files = [
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose setup"),
        (".dockerignore", "Docker ignore file"),
        ("Procfile", "Heroku configuration"),
        ("runtime.txt", "Python version for Heroku"),
        ("setup.sh", "Heroku setup script"),
        ("railway.json", "Railway configuration"),
        (".streamlit/config.toml", "Streamlit configuration"),
    ]
    
    # Documentation files
    documentation_files = [
        ("DEPLOYMENT.md", "Deployment guide"),
        ("QUICK_DEPLOY.md", "Quick deployment instructions"),
        ("deploy.sh", "Deployment script"),
    ]
    
    print("\n📋 ESSENTIAL FILES:")
    print("-" * 30)
    essential_ok = all(check_file_exists(file, desc) for file, desc in essential_files)
    
    print("\n🚀 DEPLOYMENT CONFIGURATION:")
    print("-" * 30)
    deployment_ok = all(check_file_exists(file, desc) for file, desc in deployment_files)
    
    print("\n📚 DOCUMENTATION:")
    print("-" * 30)
    docs_ok = all(check_file_exists(file, desc) for file, desc in documentation_files)
    
    # Check requirements.txt content
    print("\n🔍 REQUIREMENTS VALIDATION:")
    print("-" * 30)
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
            
        required_packages = ["streamlit", "pandas", "numpy", "plotly"]
        requirements_ok = True
        
        for package in required_packages:
            if package in requirements:
                print(f"✅ {package} found in requirements.txt")
            else:
                print(f"❌ {package} missing from requirements.txt")
                requirements_ok = False
                
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        requirements_ok = False
    
    # Check Streamlit app syntax
    print("\n🔧 APP SYNTAX CHECK:")
    print("-" * 30)
    try:
        with open("streamlit_app_simple.py", "r") as f:
            app_content = f.read()
        
        # Basic syntax check
        compile(app_content, "streamlit_app_simple.py", "exec")
        print("✅ Streamlit app syntax is valid")
        app_ok = True
        
        # Check for cyberpunk theme
        if "#00ffff" in app_content and "#1a1a1a" in app_content:
            print("✅ Cyberpunk theme colors detected")
        else:
            print("⚠️  Cyberpunk theme colors not detected")
            
    except SyntaxError as e:
        print(f"❌ Syntax error in Streamlit app: {e}")
        app_ok = False
    except FileNotFoundError:
        print("❌ streamlit_app_simple.py not found")
        app_ok = False
    
    # Git repository check
    print("\n📦 GIT REPOSITORY:")
    print("-" * 30)
    if Path(".git").exists():
        print("✅ Git repository initialized")
        git_ok = True
        
        # Check if remote is set
        try:
            import subprocess
            result = subprocess.run(["git", "remote", "-v"], 
                                  capture_output=True, text=True)
            if "github.com" in result.stdout:
                print("✅ GitHub remote configured")
                print(f"   Remote: {result.stdout.strip().split()[1]}")
            else:
                print("⚠️  GitHub remote not configured")
        except:
            print("⚠️  Could not check git remote")
    else:
        print("❌ Git repository not initialized")
        git_ok = False
    
    # Platform-specific checks
    print("\n🌐 PLATFORM CONFIGURATIONS:")
    print("-" * 30)
    
    platforms = {
        "Streamlit Cloud": ".streamlit/config.toml",
        "Heroku": "Procfile",
        "Railway": "railway.json",
        "Docker": "Dockerfile"
    }
    
    platform_ok = True
    for platform, config_file in platforms.items():
        if Path(config_file).exists():
            print(f"✅ {platform} configuration ready")
        else:
            print(f"❌ {platform} configuration missing")
            platform_ok = False
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT READINESS ASSESSMENT")
    print("=" * 60)
    
    checks = [
        ("Essential Files", essential_ok),
        ("Deployment Config", deployment_ok),
        ("Documentation", docs_ok),
        ("Requirements", requirements_ok),
        ("App Syntax", app_ok),
        ("Git Repository", git_ok),
        ("Platform Configs", platform_ok)
    ]
    
    passed_checks = sum(1 for _, status in checks if status)
    total_checks = len(checks)
    
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    success_rate = (passed_checks / total_checks) * 100
    print(f"\n🎯 Overall Readiness: {success_rate:.1f}% ({passed_checks}/{total_checks})")
    
    if success_rate >= 90:
        print("\n🎉 EXCELLENT! Ready for deployment!")
        print("🚀 You can deploy to any platform:")
        print("   • Streamlit Cloud: https://share.streamlit.io/")
        print("   • Railway: https://railway.app/")
        print("   • Heroku: Use Heroku CLI")
        print("   • Docker: Run docker-compose up -d")
    elif success_rate >= 75:
        print("\n👍 GOOD! Minor issues to address before deployment")
        print("   Review the failed checks above")
    else:
        print("\n⚠️  NEEDS ATTENTION! Several issues to resolve")
        print("   Please fix the failed checks before deploying")
    
    print(f"\n📁 Repository: https://github.com/Abhijeet-077/churn-assignment.git")
    print(f"🎮 Ready to deploy your Cyberpunk Churn Prediction app!")
    
    return success_rate >= 75

if __name__ == "__main__":
    check_deployment_readiness()
