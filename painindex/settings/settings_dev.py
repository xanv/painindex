from painindex.settings.settings_base import *


DEBUG = True
TEMPLATE_DEBUG = True

# Apps used specifically for development
INSTALLED_APPS += (

)

# Database
# https://docs.djangoproject.com/en/1.6/ref/settings/#databases
# DATABASES = {
# }

#For development, I don't actually send emails.
# This makes emails print to the console:
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Alternatively, if I wanted them to print to a text file, create
# this folder:
# EMAIL_BACKEND = 'django.core.mail.backends.filebased.EmailBackend'
# EMAIL_FILE_PATH = '/emails_testing'


# Development hosts: localhost i.e. 127.0.0.1
ALLOWED_HOSTS = ['localhost', '127.0.0.1']