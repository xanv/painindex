import os
import dj_database_url
from painindex.settings.settings_base import *

# This file is NOT part of our repo. It contains sensitive settings like secret key
# and db setup.
from env import *


DEBUG = False
TEMPLATE_DEBUG = False

# Apps used specifically for production
INSTALLED_APPS += (
    'gunicorn',
)

# These people will get error emails in production
ADMINS = (
    ('Xan', 'xan.vong@gmail.com'),
)

# Set this to match the domains of the production site.
ALLOWED_HOSTS = [
    'www.thepainindex.com', 'thepainindex.com',
    'http://still-taiga-5292.herokuapp.com',
    'localhost'
]

###################
# Heroku settings #
###################

# See https://devcenter.heroku.com/articles/getting-started-with-django

DATABASES['default'] = dj_database_url.config()
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Heroku instructions allow all hosts. 
# If I have a problem, try this.
# ALLOWED_HOSTS = ['*']

STATIC_ROOT = 'staticfiles'
STATIC_URL = '/static/'
STATICFILES_DIRS = (
    os.path.join(BASE_DIR, 'static')
)
