from django.db import models


# Create your models here.
class ScheduledActivity(models.Model):
    class Meta:
        verbose_name_plural = 'Activities'

    order_id = models.CharField(max_length=100)
    product_id = models.CharField(max_length=100)
    product_name = models.CharField(max_length=100)
    start_datetime = models.CharField(max_length=100)
    end_datetime = models.CharField(max_length=100)
    activitytype = models.CharField(max_length=1)
    status=models.CharField(max_length=1)
    created_on = models.DateTimeField(auto_now_add=True)
    last_updated_on = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.order_id)
