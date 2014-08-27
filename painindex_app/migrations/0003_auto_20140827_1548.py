# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('painindex_app', '0002_auto_20140823_2046'),
    ]

    operations = [
        migrations.CreateModel(
            name='PainReportProfile',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('user', models.OneToOneField(to=settings.AUTH_USER_MODEL)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='painreport',
            name='profile',
            field=models.ForeignKey(blank=True, to='painindex_app.PainReportProfile', null=True),
            preserve_default=True,
        ),
        migrations.RemoveField(
            model_name='painreport',
            name='pain_profile',
        ),
        migrations.DeleteModel(
            name='PainProfile',
        ),
    ]
