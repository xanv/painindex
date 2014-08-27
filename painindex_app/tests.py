from django.test import TestCase
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User
from painindex_app.models import PainTag, PainSource, PainReport, PainReportProfile

# GOAL: Have one TestClass for each model or view
# A separate test method for each set of conditions you want to test
# test method names that describe their function.


# HELPER FUNCTIONS #

def make_tags_and_sources(num_tags, num_sources):
    """ Create a generic set of PainTags and PainSources.

    They are named paintag_0, paintag_1, ...
    (similarly for painsource_0, ...).
        
    These can be defined without specifying any relationships.

    In the tests below, we follow the convention:
    T, S = populate_db()
    (That is, Tag, Source, Report)
    And thus T[0] refers to the PainTag name="paintag_0"
    """

    T = [PainTag.objects.create(name="paintag_" + str(i))
            for i in range(num_tags)]
    S = [PainSource.objects.create(name="painsource_" + str(i),)
            for i in range(num_sources)]

    return T, S

class PainSourceModelTests(TestCase):

    def test_get_rating(self):
        T, S = make_tags_and_sources(0, 2)

        R0 = PainReport.objects.create(
            pain_source=S[0],
            intensity=5
        )
        R1 = PainReport.objects.create(
            pain_source=S[0],
            intensity=7
        )

        # The rating is correctly averaged
        rating = S[0].get_rating()
        self.assertEqual(rating, 6)
        
        # The database itself contains the correct intensities
        rating = PainSource.objects.get(name="painsource_0").get_rating()
        self.assertEqual(rating, 6)

        # Noninteger averaging works correctly
        R1.intensity = 6
        R1.save()
        rating = PainSource.objects.get(name="painsource_0").get_rating()
        self.assertEqual(rating, 5.5)


