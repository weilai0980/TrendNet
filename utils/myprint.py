# -*- coding: utf-8 -*-
"""print while store the records."""
import utils.opfiles as opfile
import settings.parameters as para


def myprint(content, path=para.RECORD_DIRECTORY):
    """print the content while store the information to the path."""
    print(content)
    opfile.write_txt(content + "\n", path, type="a")
