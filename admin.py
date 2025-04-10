"""
Admin module for Visionx Ai Beginners Cryptocurrency Dashboard
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from app import db
from models import User, FeaturedCoin
import api_service
import logging
from functools import wraps

# Create blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Configure logging
logger = logging.getLogger(__name__)

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access admin panel', 'warning')
            return redirect(url_for('login'))
        
        user_id = session['user_id']
        user = User.query.get(user_id)
        
        if not user or not user.is_admin:
            flash('You do not have permission to access this page', 'danger')
            return redirect(url_for('index'))
        
        return f(*args, **kwargs)
    return decorated_function

# Admin routes
@admin_bp.route('/')
@admin_required
def admin_dashboard():
    """Admin Dashboard"""
    # Get basic stats
    users_count = User.query.count()
    featured_coins_count = FeaturedCoin.query.count()
    
    return render_template(
        'admin/dashboard.html',
        users_count=users_count,
        featured_coins_count=featured_coins_count
    )

@admin_bp.route('/featured-coins')
@admin_required
def featured_coins():
    """Manage Featured Coins"""
    coins = FeaturedCoin.query.order_by(FeaturedCoin.position).all()
    
    # Get available coins from API for adding new featured coins
    top_coins = api_service.get_top_coins(count=100)
    formatted_coins = api_service.format_market_data(top_coins)
    
    return render_template(
        'admin/featured_coins.html',
        featured_coins=coins,
        available_coins=formatted_coins
    )

@admin_bp.route('/featured-coins/add', methods=['POST'])
@admin_required
def add_featured_coin():
    """Add a new featured coin"""
    coin_id = request.form.get('coin_id')
    display_name = request.form.get('display_name')
    description = request.form.get('description', '')
    position = request.form.get('position', 0)
    
    if not coin_id or not display_name:
        flash('Coin ID and display name are required', 'danger')
        return redirect(url_for('admin.featured_coins'))
    
    # Check if coin already exists
    existing = FeaturedCoin.query.filter_by(coin_id=coin_id).first()
    if existing:
        flash(f'Coin {display_name} is already featured', 'warning')
        return redirect(url_for('admin.featured_coins'))
    
    try:
        # Get next position if not specified
        if not position or position == '0':
            max_position = db.session.query(db.func.max(FeaturedCoin.position)).scalar() or 0
            position = max_position + 10  # Leave gaps for reordering
        
        new_coin = FeaturedCoin(
            coin_id=coin_id,
            display_name=display_name,
            description=description,
            position=position,
            added_by=session['user_id']
        )
        
        db.session.add(new_coin)
        db.session.commit()
        flash(f'Coin {display_name} added to featured coins', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding featured coin: {e}")
        flash('Error adding featured coin', 'danger')
    
    return redirect(url_for('admin.featured_coins'))

@admin_bp.route('/featured-coins/update/<int:id>', methods=['POST'])
@admin_required
def update_featured_coin(id):
    """Update a featured coin"""
    coin = FeaturedCoin.query.get_or_404(id)
    
    coin.display_name = request.form.get('display_name', coin.display_name)
    coin.description = request.form.get('description', coin.description)
    coin.position = request.form.get('position', coin.position)
    coin.is_active = 'is_active' in request.form
    
    try:
        db.session.commit()
        flash(f'Coin {coin.display_name} updated', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating featured coin: {e}")
        flash('Error updating featured coin', 'danger')
    
    return redirect(url_for('admin.featured_coins'))

@admin_bp.route('/featured-coins/delete/<int:id>', methods=['POST'])
@admin_required
def delete_featured_coin(id):
    """Delete a featured coin"""
    coin = FeaturedCoin.query.get_or_404(id)
    coin_name = coin.display_name
    
    try:
        db.session.delete(coin)
        db.session.commit()
        flash(f'Coin {coin_name} removed from featured coins', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting featured coin: {e}")
        flash('Error removing featured coin', 'danger')
    
    return redirect(url_for('admin.featured_coins'))

@admin_bp.route('/featured-coins/reorder', methods=['POST'])
@admin_required
def reorder_featured_coins():
    """Reorder featured coins via AJAX"""
    try:
        # Get ordered list of coin IDs
        new_order = request.json.get('order', [])
        
        # Update positions
        for position, coin_id in enumerate(new_order):
            # Multiple by 10 to leave room for manual positioning
            FeaturedCoin.query.filter_by(id=coin_id).update({'position': position * 10})
        
        db.session.commit()
        return jsonify({'success': True, 'message': 'Order updated'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error reordering featured coins: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@admin_bp.route('/users')
@admin_required
def manage_users():
    """Manage Users"""
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=users)

@admin_bp.route('/users/toggle-admin/<int:id>', methods=['POST'])
@admin_required
def toggle_admin(id):
    """Toggle admin privileges for a user"""
    if id == session['user_id']:
        flash('You cannot change your own admin status', 'warning')
        return redirect(url_for('admin.manage_users'))
    
    user = User.query.get_or_404(id)
    
    try:
        user.is_admin = not user.is_admin
        db.session.commit()
        flash(f'Admin status for {user.username} updated', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error toggling admin status: {e}")
        flash('Error updating admin status', 'danger')
    
    return redirect(url_for('admin.manage_users'))