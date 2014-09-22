from django.contrib import admin
from django.db import models
from django import forms
from painindex_app.models import PainTag, PainSource, PainReport, PainReportProfile


admin.site.register(PainTag)
admin.site.register(PainSource)
admin.site.register(PainReport)
admin.site.register(PainReportProfile)
