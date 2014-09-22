from django.test import TestCase
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User
from painindex_app.models import PainTag, PainSource, PainReport, PainReportProfile

# GOAL: Test model methods and views.
# Have one TestClass for each (nontrivial) model and view,
# with a separate test method for each set of conditions you want to test.
# Test method names should describe their function.

####################
# HELPER FUNCTIONS #
####################

def make_tags_and_sources(num_tags, num_sources):
    """ Create a generic set of PainTags and PainSources.

    They are named paintag_0, paintag_1, ...
    (similarly for painsource_0, ...).
        
    These can be defined without specifying any relationships.

    In the tests below, we follow the convention:
    T, S = populate_db()
    (That is, Tag, Source)
    And thus T[0] refers to the PainTag name="paintag_0"
    """

    T = [PainTag.objects.create(name="paintag_" + str(i))
            for i in range(num_tags)]
    S = [PainSource.objects.create(name="painsource_" + str(i),)
            for i in range(num_sources)]

    return T, S

###############
# MODEL TESTS #
###############

class PainSourceManagerTests(TestCase):

    def test_select_random_in_range(self):
        T, S = make_tags_and_sources(0, 3)
        R0 = PainReport.objects.create(
            pain_source=S[0],
            intensity=5
        )
        R1 = PainReport.objects.create(
            pain_source=S[0],
            intensity=5
        )

        S[0].calc_rating()

        # Correctly selects the unique source in the range.
        mysource = PainSource.objects.select_random_in_range(4.5, 5.5)
        self.assertEqual(mysource.name, "painsource_0")

        # Returns None when nothing is in the range
        mysource = PainSource.objects.select_random_in_range(5.1, 5.5)
        self.assertEqual(mysource, None)

        # Return None when range is empty
        mysource = PainSource.objects.select_random_in_range(5.5, 4.5)
        self.assertEqual(mysource, None)
        
        
        R2 = PainReport.objects.create(
            pain_source=S[1],
            intensity=5
        )
        S[1].calc_rating()
        R3 = PainReport.objects.create(
            pain_source=S[2],
            intensity=6
        )
        S[2].calc_rating()

        # Select randomly when multiple sources are in range
        # Warning: This is not a deterministic test.
        mysource = PainSource.objects.select_random_in_range(4.5, 5.5)
        assert mysource.name in ['painsource_0', 'painsource_1']


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
        rating = S[0].calc_rating()
        self.assertEqual(rating, 6)
        
        # The correct rating has been saved to the db by calc_rating
        rating = PainSource.objects.get(name="painsource_0").pain_rating
        self.assertEqual(rating, 6)

        # Noninteger averaging works correctly
        R1.intensity = 6
        R1.save()
        rating = PainSource.objects.get(name="painsource_0").calc_rating()
        self.assertEqual(rating, 5.5)

        # calc_rating returns None if no PainReport.
        rating = S[1].calc_rating()
        self.assertEqual(rating, None)

        # No pain_rating is saved to the db if no PainReport.
        # (A NULL field in db corresponds to None in Django).
        rating = PainSource.objects.get(name="painsource_1").pain_rating
        self.assertEqual(rating, None)

    def test_reviews(self):
        "TODO..."
        # Exercise for CFV: Refactor the reviews method of PainSource
        # to use a list comprehension 
        pass
    def test_short_description(self):
        "TODO...be sure to test the case where description is None"
        pass



###############
# VIEWS TESTS #
###############

class HomepageViewTests(TestCase):

    def test_homepage_view_with_no_data(self):
        "Ensure the homepage loads even without any data."
        response = self.client.get(reverse('painindex_app:homepage'))
        self.assertEqual(response.status_code, 200)        

    def test_homepage_view_with_sparse_data(self):
        T, S = make_tags_and_sources(0, 2)
        R0 = PainReport.objects.create(
            pain_source=S[0],
            intensity=5
        )
        R1 = PainReport.objects.create(
            pain_source=S[0],
            intensity=7
        )
        S[0].calc_rating()

        response = self.client.get(reverse('painindex_app:homepage'))
        # See all available attributes and methods of response:
        # print dir(response)

        self.assertEqual(response.status_code, 200)

        self.assertEqual(response.context['highlighted_bug'], S[0])
        assert S[0] in response.context['find_bugs']
        self.assertEqual( len(response.context['find_bugs']), 10 )
        

        self.assertContains(response, "The Pain Index")
        self.assertContains(response, "painsource_0")
        # Unrated source should not be displayed
        self.assertNotContains(response, "painsource_1")

    def test_homepage_view_with_more_data(self):
        "TODO"
        # test what happens when there are multiple painsources with same
        # ratings, etc.
        pass

    # Can also test other elements of homepage...

class PainSourceDetailViewTests(TestCase):
    "TODO"
    pass

class PainReportViewTests(TestCase):
    "TODO"
    pass
    # will want to test that posted info is reflected in the updated pain index.