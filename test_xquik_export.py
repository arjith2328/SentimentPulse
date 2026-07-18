import unittest

from dashboard.xquik_export import normalize_xquik_records


class XquikExportTests(unittest.TestCase):
    def test_normalizes_export_rows_for_dashboard_columns(self):
        rows = normalize_xquik_records(
            [
                {
                    "tweet_id": "42",
                    "full_text": "Great product update",
                    "author_username": "builder",
                    "created_at": "2026-07-04",
                    "label": "positive",
                }
            ]
        )

        self.assertEqual(
            rows,
            [
                {
                    "text": "Great product update",
                    "cleaned_text": "great product update",
                    "user": "builder",
                    "date": "2026-07-04",
                    "sentiment": "positive",
                    "source_id": "42",
                }
            ],
        )

    def test_skips_blank_rows_and_defaults_sentiment(self):
        rows = normalize_xquik_records(
            [
                {"text": ""},
                {"message": "Needs review"},
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sentiment"], "neutral")

    def test_matches_columns_case_insensitively_without_fake_dates(self):
        rows = normalize_xquik_records(
            [
                {
                    "Tweet_Text": "Mixed Case",
                    "Author_Username": "analyst",
                    "Created_At": float("nan"),
                }
            ]
        )

        self.assertEqual(rows[0]["text"], "Mixed Case")
        self.assertEqual(rows[0]["user"], "analyst")
        self.assertEqual(rows[0]["date"], "")


if __name__ == "__main__":
    unittest.main()
