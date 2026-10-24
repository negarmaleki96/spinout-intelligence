import re

def test_doi_normalization_pattern():
    result = re.sub(r"^https?://(dx\\.)?doi\\.org/", "", "https://doi.org/10.1234/ABC".lower())
    assert result == "10.1234/abc"
