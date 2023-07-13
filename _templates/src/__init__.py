import datetime

name = "<<name>>"
package_name = "<<package-name>>"
author = "<<author>>"
author_email = "<<email>>"
description = "<<description>>"
url = "<<url>>"
project_urls = {
    <<requires::docs "Documentation": "https://<<organization>>.github.io/<<name>>/main", endrequires::docs>>
    "Source Code": "https://github.com/<<organization>>/<<package-name>>",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, <<author>>"
version = "0.0.1"
