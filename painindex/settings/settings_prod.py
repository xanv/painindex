import os
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


# Configure production emails.


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

# Define place my static files will be collected and served from.
# See https://docs.djangoproject.com/en/1.6/ref/settings/#std:setting-STATIC_ROOT
# STATIC_ROOT = ""

# MEDIA_ROOT = ""