# Generated by Django 5.1.6 on 2025-06-28 18:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("smartassess", "0028_notification"),
    ]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="date_joined",
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
