import pandas as pd
import re


def clean_location(df):
    # Expanded list of U.S. states with abbreviations
    us_states = {
        "AL": "Alabama",
        "AK": "Alaska",
        "AZ": "Arizona",
        "AR": "Arkansas",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DE": "Delaware",
        "FL": "Florida",
        "GA": "Georgia",
        "HI": "Hawaii",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "IA": "Iowa",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "ME": "Maine",
        "MD": "Maryland",
        "MA": "Massachusetts",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MS": "Mississippi",
        "MO": "Missouri",
        "MT": "Montana",
        "NE": "Nebraska",
        "NV": "Nevada",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NY": "New York",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PA": "Pennsylvania",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VT": "Vermont",
        "VA": "Virginia",
        "WA": "Washington",
        "WV": "West Virginia",
        "WI": "Wisconsin",
        "WY": "Wyoming",
    }

    # Expanded list of countries with variations
    countries = {
        "USA": ["United States", "United States of America", "America", "US", "U.S.", "U.S.A."],
        "UK": ["United Kingdom", "Britain", "Great Britain", "England"],
        "China": ["PRC", "People's Republic of China", "P. R. China"],
        "Russia": ["Russian Federation"],
        "South Korea": ["Korea"],
        "North Korea": ["DPRK"],
        "Iran": ["Persia"],
        # Add more countries and their variations as needed
    }

    # Flatten the countries dictionary for easier lookup
    flat_countries = {alias.lower(): country for country, aliases in countries.items() for alias in aliases}
    for country in countries.keys():
        flat_countries[country.lower()] = country

    # Add more countries from the dataset
    additional_countries = [
        "Germany",
        "Italy",
        "France",
        "Spain",
        "Canada",
        "Mexico",
        "Brazil",
        "Argentina",
        "Japan",
        "India",
        "Australia",
        "New Zealand",
        "Netherlands",
        "Belgium",
        "Sweden",
        "Norway",
        "Denmark",
        "Finland",
        "Switzerland",
        "Austria",
        "Poland",
        "Greece",
        "Turkey",
        "Egypt",
        "South Africa",
        "Nigeria",
        "Kenya",
        "Saudi Arabia",
        "UAE",
        "Israel",
        "Iran",
        "Pakistan",
        "Bangladesh",
        "Thailand",
        "Vietnam",
        "Indonesia",
        "Philippines",
        "Malaysia",
        "Singapore",
        "Albania",
        "Azerbaijan",
        "Bulgaria",
        "Cameroon",
        "Chile",
        "Colombia",
        "Costa Rica",
        "Czech Republic",
        "Hong Kong",
        "Iceland",
        "Luxembourg",
        "Nepal",
        "Panama",
        "Peru",
        "Portugal",
        "Puerto Rico",
        "Romania",
        "Taiwan",
        "Uruguay",
        "Venezuela",
        "Yugoslavia",
        "Uzbekistan",
        "Hong Kong",
    ]

    for country in additional_countries:
        if country.lower() not in flat_countries:
            flat_countries[country.lower()] = country

    # Create a set of all state names (both full names and abbreviations)
    all_states = set(us_states.keys()) | set(us_states.values())

    # Dictionary of city to state mappings
    city_to_state = {
        "Chicago": "Illinois",
        "Miami": "Florida",
        "Los Angeles": "California",
        "San Francisco": "California",
        "Philadelphia": "Pennsylvania",
        "Houston": "Texas",
        "Brooklyn": "New York",
        "Ann Arbor": "Michigan",
        "Staten Island": "New York",
        "Minneapolis": "Minnesota",
        "Cleveland": "Ohio",
        "Detroit": "Michigan",
        "Atlanta": "Georgia",
        "Berkeley": "California",
        "Baltimore": "Maryland",
        "Pittsburgh": "Pennsylvania",
        "Seattle": "Washington",
        "Manhattan": "New York",
        "Washington": "District of Columbia",
        "Boston": "Massachusetts",
        "San Diego": "California",
        "Salt Lake City": "Utah",
        "New York": "New York",
        "New York City": "New York",
        "Sofia": "Bulgaria",
        "London": "UK",
        "Toronto": "Canada",
        "Milan": "Italy",
        "Pougkeepsie": "New York",
        "Vestal": "New York",
        "Katonah": "New York",
        "Bronx": "New York",
        "Paris": "France",
        "Tokyo": "Japan",
        "Long Island": "New York",
        "Siberia": "Russia",
        "Westchester County": "New York",
        "Washington DC": "WA",
        "Salt Lake City": "Utah",
    }

    # Dictionary for replacing location phrases
    location_replacements = {
        "Wash DC (4 yrs)": "Washington",
        "California (West Coast)": "California",
        "NYC (Staten Island)": "Staten Island",
        "Washington DC Metro Region": "Washington",
        "I am from NYC": "New York City",
        "NYC-6 yrs. Grew up in Nebraska": "New York City",
        "way too little space here. world citizen.": "Unknown",
        "Tokyo and Texas": "Tokyo",
        "USA/American": "USA",
        "Midwest USA": "USA",
        "San Francisco Bay Area": "San Francisco",
        "Detroit suburbs": "Detroit",
        "Upstate New York": "New York",
        "Born in Iran": "Iran",
        "San Francisco(home)/Los Angeles(undergrad)": "San Francisco",
        "NYC": "New York",
        "94115": "San Francisco",
        "brooklyn ny": "Brooklyn",
        "Washington State": "WA",
        "Southern California": "California",
        "Northern California": "California",
        "Bowdoin College": "Unknown",
        "Milan - Italy": "Milan",
        "Pougkeepsie NY": "Pougkeepsie",
        "Northern Virginia": "Virginia",
        "J.P. Morgan": "Unknown",
        "International Student": "Unknown",
        "California and New York": "California",
        "Tokyo and Texas": "Tokyo",
        "Bronx Science": "Bronx",
        "Midwest USA": "USA",
        "Florida and Virginia": "Florida",
        "Hawaii and Los Angeles": "Hawaii",
        "working": "Unknown",
        "HKG": "Hong Kong",
        "Westchester County, N.Y.": "Westchester County",
        "Washington DC": "Washington",
        "DC": "Washington",
    }

    # Function to normalize location names
    def normalize_location(location):
        # Trim leading/trailing whitespaces and convert to lowercase
        location = location.strip().lower()

        # Check if the location is in the replacement dictionary (case-insensitive)
        for key, value in location_replacements.items():
            if location == key.strip().lower():
                return value.title()  # Return the replacement value with proper capitalization

        # If location contains a separator, take only the first part
        if "/" in location or "&" in location:
            parts = re.split(r"[/&]", location)
            normalized_parts = [normalize_location(part.strip()) for part in parts]
            # Filter out 'Unknown' values
            valid_parts = [part for part in normalized_parts if part != "Unknown"]
            if valid_parts:
                return valid_parts[0]  # Return the first valid part
            else:
                return "Unknown"

        if location in flat_countries:
            return flat_countries[location].title()
        for abbr, full_name in us_states.items():
            if location == abbr.lower() or location == full_name.lower():
                return full_name.title()

        # Capitalize each word for cities, states, and countries
        return " ".join(word.capitalize() for word in location.split())

    #  # Create a dictionary of zipcodes from the dataset
    # zipcodes = set(df['zipcode'].dropna().unique())
    # zipcode_dict = {zipcode: 'USA' if len(str(zipcode)) == 5 else 'Unknown' for zipcode in zipcodes}

    # Function to extract city and state_or_country
    def extract_location(row):
        from_val = str(row["from"]).strip()
        undergra_val = str(row["undergra"])
        # zipcode_val = str(row['zipcode'])

        city = "Unknown"
        state_or_country = "Unknown"

        # Normalize the 'from' value
        normalized_from = normalize_location(from_val)

        # if normalized_from in flat_countries.values(): # mii
        if normalized_from.lower() in (value.lower() for value in flat_countries.values()):
            state_or_country = normalized_from
        elif normalized_from in us_states.values():
            state_or_country = normalized_from
        elif "," in normalized_from:
            parts = [normalize_location(part) for part in normalized_from.split(",")]
            city = parts[0]
            if len(parts) > 1:
                state_or_country = parts[1]
                if state_or_country in us_states.values():
                    pass  # Keep the state name as is
                elif state_or_country in flat_countries.values():
                    pass  # Keep the country name as is
                else:
                    state_or_country = "Unknown"
        else:
            city = normalized_from

        # Check if the city is in our city_to_state dictionary
        if city in city_to_state:
            state_or_country = city_to_state[city]

        # Handle cities that are also country names
        if city in flat_countries.values():
            state_or_country = city
            city = "Unknown"

        # Check undergraduate information if city is Unknown
        if city == "Unknown" and "University" in undergra_val:
            match = re.search(r"University of (.+)", undergra_val)
            if match:
                city = normalize_location(match.group(1))

        # # Use zipcode information if available and state_or_country is still Unknown
        # if state_or_country == 'Unknown' and zipcode_val in zipcode_dict:
        #     state_or_country = zipcode_dict[zipcode_val]

        # Final check: if city is a state, move it to state_or_country
        if city in us_states.values():
            state_or_country = city
            city = "Unknown"

        return pd.Series({"city": city, "state_or_country": state_or_country})

    # Apply the function to create new columns
    df[["city", "state_or_country"]] = df.apply(extract_location, axis=1)

    return df


def clean_university_names(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Convert to lowercase
    df["undergra"] = df["undergra"].str.lower()

    # Remove leading/trailing whitespace
    df["undergra"] = df["undergra"].str.strip()

    # Standardize common abbreviations
    abbrev_dict = {
        r"\bu\b": "university",
        r"\buniv\b": "university",
        r"\bcoll\b": "college",
        r"\bst\b": "saint",
    }
    for abbrev, full in abbrev_dict.items():
        df["undergra"] = df["undergra"].str.replace(abbrev, full, regex=True)

    # Remove punctuation
    df["undergra"] = df["undergra"].str.replace(r"[^\w\s]", "", regex=True)

    # Handle special characters
    df["undergra"] = df["undergra"].str.replace("&", "and")

    # Standardize common variations
    variations_dict = {
        "uc ": "university of california ",
        "u of ": "university of ",
    }
    for var, standard in variations_dict.items():
        df["undergra"] = df["undergra"].str.replace(var, standard)

    # Remove redundant words
    # redundant_words = ['of', 'the']
    # pattern = r'\b(?:{})\b'.format('|'.join(redundant_words))
    # df['undergra'] = df['undergra'].str.replace(pattern, '', regex=True)

    # Group similar names (example for a few universities)
    groupings = {
        "harvard": ["harvard", "harvard university", "harvard college"],
        "mit": ["mit", "massachusetts institute of technology"],
        "stanford": ["stanford", "stanford university"],
        "columbia": ["columbia", "columbia university", "columbia college"],
    }
    for standard, variations in groupings.items():
        df.loc[df["undergra"].isin(variations), "undergra"] = standard

    # Final cleanup: remove extra spaces
    df["undergra"] = df["undergra"].str.replace(r"\s+", " ", regex=True).str.strip()

    return df


# MII_REF17: It is better this f-n to return a copy of the df
def clean_field_of_study(df, col):
    def clean_field(field):
        # Convert to lowercase
        field = field.lower()

        # Take part before separators
        separators = r"[-/:\(&/,]"
        field = re.split(separators, field)[0].strip()

        # Rename specific fields

        categories = {
            "mathematics": ["math", "mathematics"],
            "education": ["education", "teaching", "educational", "educator", "teaching", "teacher", "professor"],
            "business": ["business"],
            "art": ["art", "arts"],
            "biology": ["biology"],
            "psychology": ["psychology", "psychologist"],
            "writing": ["writing", "poet", "writer", "literature"],
            "biomedical": ["biomedical", "biotechnology", "biotech"],
            "economics": ["economics", "economist", "economic"],
            "engineering": ["engineering", "engg", "engineer"],
            "english": ["english"],
            "finance": ["finance", "financial", "finanace", "financing", "equity"],
            "history": ["history", "historian"],
            "law": ["law", "attorney"],
            "medicine": ["medicine", "medical", "doctor", "cardiologist", "clinic", "epidemiologist", "physician"],
            "nutrition": ["nutrition", "nutritiron"],
            "philosophy": ["philosophy"],
            "theater": ["theater", "theatre"],
            "climate": ["climate"],
            "management": ["management"],
            "biotechnology": ["biotechnology"],
            "investment": ["investment"],
            "undecided": ["know", "?", "??", "idea", "knew", "sure", "wondering"],
            "academics": ["academia", "academics", "academic"],
            "speech": ["speech"],
            "trading": ["trader", "trading"],
            "entrepreneurship": ["entrepeneur", "entrepreneur", "entrepreneurship"],
            "film": ["filmmaker", "film"],
            "health": ["health"],
            "journalism": ["journalist", "journalism"],
            "marketing": ["marketing"],
            "music": ["music"],
            "consulting": ["consulting", "consultant"],
            "banking": ["banking", "banker"],
            "research": ["research"],
            "social work": ["social"],
            "mfa": ["mfa"],
            "social": ["social"],
            "science": ["bio", "chemistry", "physics", "neuroscience", "neurosciences", "scientist", "science"],
        }

        # Remove "master of", "masters of", "master in", "masters in"
        field = re.sub(r"master(?:s)? (?:of|in)\s*", "", field).strip()

        for category, keywords in categories.items():
            if any(keyword in field for keyword in keywords):
                field = category

        return field

    # Apply the cleaning function to the 'field' column
    df[col] = df[col].apply(clean_field)

    return df


# Example usage:
# df = pd.read_csv('your_data.csv')
# df = clean_field_column(df)
# print(df['field'])
