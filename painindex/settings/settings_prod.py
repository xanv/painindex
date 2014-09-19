from painindex.settings.settings_base import *
from env import *

DEBUG = False
TEMPLATE_DEBUG = False

# Apps used specifically for production
INSTALLED_APPS += (

)


# Configure production emails.




# Set this to match the domains of the production site.
# ALLOWED_HOSTS = []

# Define place my static files will be collected and served from.
# See https://docs.djangoproject.com/en/1.6/ref/settings/#std:setting-STATIC_ROOT
# STATIC_ROOT = ""