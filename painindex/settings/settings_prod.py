import os
import dj_database_url
from painindex.settings.settings_base import *


try:
    # This file is not part of the repo and contains secrets like db info.
    from env import *
# There is no env.py file on Heroku.
# We load from environment variables instead.
except ImportError:
    SECRET_KEY = os.environ['PAIN_INDEX_SECRET_KEY']

    EMAIL_HOST = 'smtp.gmail.com'
    EMAIL_PORT = 587
    EMAIL_HOST_PASSWORD = os.environ['PAIN_INDEX_EMAIL_HOST_PASSWORD']
    EMAIL_HOST_USER = os.environ['PAIN_INDEX_EMAIL_HOST_USER']
    EMAIL_USE_TLS = True


DEBUG = False
TEMPLATE_DEBUG = False

# Apps used specifically for production
INSTALLED_APPS += (
    'gunicorn',
)

# These people will get error emails in production
# ADMINS = (
#     ('Xan', 'xan.vong@gmail.com'),
# )

# Set this to match the domains of the production site.
ALLOWED_HOSTS = [
    'www.thepainindex.com', 'thepainindex.com',
    'http://still-taiga-5292.herokuapp.com',
    # 'still-taiga-5292.herokuapp.com/',
    'localhost'
]

###################
# Heroku settings #
###################

# See https://devcenter.heroku.com/articles/getting-started-with-django

# This DATABASES default is just for running foreman locally.
# I import it in env instead. 
# DATABASES['default'] = dj_database_url.config()
# DATABASES = {'default': dj_database_url.config(default='postgres://localhost')}
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Heroku instructions say to allow all hosts. Sounds like a bad idea.
ALLOWED_HOSTS = ['*']

STATIC_ROOT = 'staticfiles' # Static files are collected here
# STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles') # Static files are collected here
STATIC_URL = '/static/'
STATICFILES_DIRS = (
    os.path.join(BASE_DIR, 'painindex/static'),
)
