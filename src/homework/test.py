from text_processing import regular_expressions

tests = [
    # ----------------
    # EMAIL
    # ----------------
    ("john@emory.edu", "email"),
    ("john.doe_1-test@host-name.org", "email"),
    ("a@b.com", "email"),
    ("a.@b.com", None),
    (".a@b.com", None),  
    ("a@-b.com", None),    
    ("a@b-.com", None),     
    ("a@b.xyz", None),   
    ("a@@b.com", None),

    # ----------------
    # URL
    # ----------------
    ("http://example.com", "url"),
    ("https://a.b", "url"),
    ("https://sub-domain.site.edu", "url"), 
    ("ftp://example.com", None), 
    ("http://-bad.com", None),
    ("http://a", None), 
    ("http://a..b", None),
    ("http://a.b.", None), 

    # ----------------
    # DATE
    # ----------------
    ("1951-1-1", "date"),
    ("51/1/1", "date"),
    ("2050-12-31", "date"),
    ("50-12-31", "date"), 
    ("00-1-1", "date"),  
    ("2051-01-01", None),   
    ("1950-12-31", None), 
    ("1951-13-1", None),  
    ("1951-4-31", None),
    ("2000-2-29", "date"),  
    ("2001-2-29", None),  

    # ----------------
    # CITE
    # ----------------
    ("Smith, 2024", "cite"),
    ("Smith and Jones, 2020", "cite"),
    ("Smith et al., 2019", "cite"),
    ("smith, 2020", None),   
    ("Smith, 1899", None),  
    ("Smith, 2025", None),   
    ("Van Helsing, 2020", "cite"),
    ("Smith-Jones, 2020", "cite"), 
]

for s, gold in tests:
    pred = regular_expressions(s)
    ok = (pred == gold)
    print(f"{s:30} -> {str(pred):5}  {'OK' if ok else 'WRONG expected ' + str(gold)}")

