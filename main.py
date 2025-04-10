from app import app
import routes  # This import is necessary to register the routes

# Import and register admin blueprint
from admin import admin_bp
app.register_blueprint(admin_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
