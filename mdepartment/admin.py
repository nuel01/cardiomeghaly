from django.contrib import admin
from .models import Mdepartments

# Register your models here.
class MdepartmentAdmin(admin.ModelAdmin):
    list_display=('title','description')
admin.site.register(Mdepartments)