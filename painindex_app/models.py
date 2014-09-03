from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator




class PainTag(models.Model):
    """ A tag to indicate the type of pain.
        For example, a sting.

        Each Pain object can have multiple tags.
    """
    name = models.CharField(max_length=100, db_index=True)

    def __unicode__(self):
        return self.name
        


class PainSource(models.Model):
    """ Model that represents a source of pain.
        For example, a yellow jacket sting.
    """
    name = models.CharField(max_length=255, unique=True)
    tags = models.ManyToManyField(PainTag, null=True, blank=True)
    
    def __unicode__(self):
        return self.name
        
    class Meta:
        ordering = ('name',)

    def get_rating(self):
        """ Get the average intensity rating from all PainReports
            associated with this PainSource.

            This is the rating that is displayed to users.
            The implementation details of how this rating is computed
            from the raw data will likely change in the future.
        """
        reports = self.painreport_set.all()
        intensities = [r.intensity for r in reports]
        avg = float(sum(intensities)) / len(reports)

        return avg




class PainReport(models.Model):
    """ A user report for a source of pain.
    """
    SCALE_MAX = 10
    SCALE = range(1,SCALE_MAX + 1)
    SCALE_CHOICES = zip(SCALE, SCALE)

    pain_source = models.ForeignKey(PainSource)
    intensity = models.IntegerField(choices=SCALE_CHOICES)
    profile = models.ForeignKey('PainReportProfile', null=True, blank=True)

    def __unicode__(self):
        return "PainReport %d: %s" % (self.pk, self.pain_source.name)

class PainReportProfile(models.Model):
    """ A bundle of all PainReports from a single entity.
        Each profile either belongs to a User or corresponds to a session
        of an anonymous user.
    """

    user = models.OneToOneField(User)
    
    def __unicode__(self):
        return "PainReportProfile %d" % self.pk
