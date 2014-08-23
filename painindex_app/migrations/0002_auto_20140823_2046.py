# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('painindex_app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='PainProfile',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='painreport',
            name='pain_profile',
            field=models.ForeignKey(blank=True, to='painindex_app.PainProfile', null=True),
            preserve_default=True,
        ),
        migrations.AlterField(
            model_name='painreport',
            name='intensity',
            field=models.IntegerField(choices=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]),
        ),
    ]
