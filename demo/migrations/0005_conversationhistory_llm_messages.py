# Generated by Django 4.2.5 on 2023-09-25 06:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0004_rename_llmmessages_message'),
    ]

    operations = [
        migrations.CreateModel(
            name='ConversationHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=255, unique=True)),
                ('messages', models.TextField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('extra_responses', models.TextField()),
                ('all_questions', models.TextField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='llm_messages',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=255, unique=True)),
                ('messages', models.TextField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
