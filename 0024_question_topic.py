# Generated by Django 5.1.6 on 2025-05-01 13:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('smartassess', '0023_studentanswer_ai_feedback_studentanswer_is_correct_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='topic',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
