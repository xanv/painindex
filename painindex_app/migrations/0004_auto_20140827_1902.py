# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('painindex_app', '0003_auto_20140827_1548'),
    ]

    operations = [
        migrations.AlterField(
            model_name='painsource',
            name='tags',
            field=models.ManyToManyField(to=b'painindex_app.PainTag', null=True, blank=True),
        ),
    ]
