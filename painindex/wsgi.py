"""
WSGI config for painindex project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/dev/howto/deployment/wsgi/
"""

import os
from dj_static import Cling # heroku staticfiles collection

os.environ["DJANGO_SETTINGS_MODULE"] = "painindex.settings.settings_prod"

from django.core.wsgi import get_wsgi_application
application = Cling(get_wsgi_application())
