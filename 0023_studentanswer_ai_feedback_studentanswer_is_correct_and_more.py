# Generated by Django 5.1.6 on 2025-04-28 13:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('smartassess', '0022_rename_out_of_practiceresult_total_score_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='studentanswer',
            name='ai_feedback',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='studentanswer',
            name='is_correct',
            field=models.BooleanField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='studentanswer',
            name='suggested_topics',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='testattempt',
            name='correct_answers',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='testattempt',
            name='score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='testattempt',
            name='suggested_topics',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='testattempt',
            name='total_questions',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='testattempt',
            name='start_time',
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterUniqueTogether(
            name='studentanswer',
            unique_together={('attempt', 'question')},
        ),
        migrations.AlterUniqueTogether(
            name='testattempt',
            unique_together={('student', 'test')},
        ),
    ]
