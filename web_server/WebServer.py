from flask import Flask, render_template, request, redirect, flash, copy_current_request_context
from flask_login import LoginManager, UserMixin, login_user, login_required
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from Requests import QuestionRequest, ArticleRequest

# ========== Website backend configuration ==========

app = Flask(__name__)
app.config['SECRET_KEY'] = '37806fa4a1c653c133874fc9e3ef576ddc83fbe294fc9f8ab97010421784eacf'

# database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
db = SQLAlchemy()


class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    isAdmin = db.Column(db.Boolean, default=False, nullable=False)


db.init_app(app)

with app.app_context():
    db.create_all()

# websocket support
socketio = SocketIO(app)

# login manager configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)


# global objects
request_dispatcher = None
compute_server_manager = None


# ========== Route definitions ==========


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handles login attempts.

    When someone attempts to log into the website via /login, the requests will be routed to this function. The form's information is available via the request variable.
    """
    if request.method == 'POST':
        form_username = request.form['username']
        form_password = request.form['password']

        user = Users.query.filter_by(username=form_username).first()

        if user is not None and user.password == form_password:
            login_user(user)
            flash('logged in!')
            return redirect('index')
        else:
            flash('incorrect credentials')

    return render_template('login.html')


@app.route('/')
@app.route('/index')
@login_required
def index():
    return render_template('index.html')


@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')


@app.route('/article-generator')
def article_generator():
    return render_template('article_generator.html')


@app.route('/compute-server')
@login_required
def compute_server():
    # a list of current compute machines is passed to the template
    return render_template('compute_server.html', server_list=compute_server_manager.compute_machines)


# ========== Web socket traffic ==========


@socketio.on('generate_chatbot_response')
@login_required
def generate_chatbot_response(data):
    """
    Handles websocket requests to generate a chatbot response.

    Tokens are returned to the client one by one until an end-of-text token is received.

    :param data: contains the request information.
    """
    sid = request.sid

    r = QuestionRequest(data['prompt'], float(data['temp']), float(data['top_p']))

    @copy_current_request_context
    def handle_token(token):
        emit('response', {'token': token})
        if token == '</s>':
            request_dispatcher.remove_request(sid)

    r.set_token_handler(handle_token)

    request_dispatcher.add_request(sid, r)


@socketio.on('generate_article')
@login_required
def generate_article(data):
    """
    Handles websocket requests to generate articles.

    Tokens are returned to the client one by one until an end-of-text token is received.

    :param data: contains the request information.
    """
    sid = request.sid

    r = ArticleRequest(data['prompt'], float(data['temp']), float(data['top_p']))

    @copy_current_request_context
    def handle_token(token):
        emit('response', {'token': token})
        if token == '</s>':
            request_dispatcher.remove_request(sid)

    r.set_token_handler(handle_token)

    request_dispatcher.add_request(sid, r)


@socketio.on('stop_generating')
@login_required
def stop_generating(data):
    """
    Handles websocket requests to stop generating a result.
    """
    request_dispatcher.remove_request(request.sid)


def run(request_dispatcher_param, compute_server_manager_param):
    global request_dispatcher, compute_server_manager
    request_dispatcher = request_dispatcher_param
    compute_server_manager = compute_server_manager_param
    app.run(host='0.0.0.0')

    """user = Users(username='admin',
                 password='iamtheadmin',
                 isAdmin=True)

    db.session.add(user)
    db.session.commit()"""
