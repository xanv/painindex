# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('painindex_app', '0005_auto_20140911_0009'),
    ]

    operations = [
        migrations.AddField(
            model_name='painsource',
            name='predicted_pain_rating',
            field=models.FloatField(null=True, blank=True),
            preserve_default=True,
        ),
    ]
