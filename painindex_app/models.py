import random
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
        
class PainSourceManager(models.Manager):
    def select_random_in_range(self, lower_bound, upper_bound):
        results = self.filter(pain_rating__gte=lower_bound).filter(pain_rating__lt=upper_bound)
        
        try:
            return random.choice(results)
        except IndexError:
            return None



class PainSource(models.Model):
    """ Model that represents a source of pain.
        For example, a yellow jacket sting.
    """
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(null=True, blank=True)
    pain_rating = models.FloatField(null=True, blank=True)
    predicted_pain_rating = models.FloatField(null=True, blank=True)
    tags = models.ManyToManyField(PainTag, null=True, blank=True)
    objects = PainSourceManager()

    def __unicode__(self):
        return self.name
        
    class Meta:
        ordering = ('name',)

    def calc_rating(self):
        """ Get the average intensity rating from all PainReports
            associated with this PainSource.

            This is the rating that is displayed to users.
            The implementation details of how this rating is computed
            from the raw data will likely change in the future.
        """
        reports = self.painreport_set.all()
        intensities = [r.intensity for r in reports]

        if len(reports) == 0:
            return None

        avg = float(sum(intensities)) / len(reports)

        self.pain_rating = avg
        self.save()
        return avg

    def reviews(self):
        """ Get all the reviews users have submitted for this PainSource"""
        reviews = []

        for report in self.painreport_set.all():
            reviews.append(report.description)
        return reviews

    def short_description(self):
        return self.description[:1000] if self.description is not None else ''


class PainReport(models.Model):
    """ A user report for a source of pain.
    """
    SCALE_MAX = 10
    SCALE = range(1,SCALE_MAX + 1)
    SCALE_CHOICES = zip(SCALE, SCALE)

    pain_source = models.ForeignKey(PainSource)
    intensity = models.IntegerField(choices=SCALE_CHOICES)
    description = models.TextField(null=True, blank=True)
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

class FunFact(models.Model):
    """A fun fact about venomous animals"""

    content = models.TextField()

    def __unicode(self):
        return "FunFact %d" % self.pk
