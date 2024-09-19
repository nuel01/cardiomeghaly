from django.db import models

# Create your models here.
class Mdepartments(models.Model):
    title=models.CharField(max_length=200)
    description=models.TextField(blank=True)
    photo_main=models.ImageField(upload_to='photos/%Y/%m/%d')
    def __str__(self):
        return self.title