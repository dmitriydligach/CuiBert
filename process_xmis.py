#!/usr/bin/env python3

# This is a Python reimplementation of
# ctakes-misc/src/main/java/org/apache/ctakes/consumers/ExtractCuiSequences.java

import os
import pathlib

from cassis import *

xmi_dir = '/Users/Dima/Work/Data/MimicIII/Notes/Xmi/'
cui_dir = '/Users/Dima/Work/Data/MimicIII/Notes/Cui/'

ident_annot_class_name = 'org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation'
umls_concept_class_name = 'org_apache_ctakes_typesystem_type_refsem_UmlsConcept'
type_system_path = 'TypeSystem.xml'

def get_ontology_concept_codes(identified_annot):
  """Extract CUIs from an identified annotation"""

  # often the same CUI added multiple times
  # must count it only once
  codes = set()

  ontology_concept_arr = identified_annot['ontologyConceptArr']
  if not ontology_concept_arr:
    # not a umls entity, e.g. fraction annotation
    return codes

  for ontology_concept in ontology_concept_arr.elements:
    if type(ontology_concept).__name__ == umls_concept_class_name:
      code = ontology_concept['cui']
      codes.add(code)
    else:
      print('WEIRDNESS')

  return codes

def process_xmi_file(xmi_path):
  """This is a Python staple"""

  type_system_file = open(type_system_path, 'rb')
  type_system = load_typesystem(type_system_file)

  xmi_file = open(xmi_path, 'rb')
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
  sys_view = cas.get_view('_InitialView')

  cuis = []
  for ident_annot in sys_view.select(ident_annot_class_name):
    # text = ident_annot.get_covered_text().replace('\n', '')
    for code in get_ontology_concept_codes(ident_annot):
      cuis.append(code)

  cuis_as_str = ' '.join(cuis)
  out_file_name = pathlib.Path(xmi_path).stem
  out_file = pathlib.Path(os.path.join(cui_dir, out_file_name))
  out_file.write_text(cuis_as_str)

if __name__ == "__main__":

  for file_name in os.listdir(xmi_dir):
    xmi_path = os.path.join(xmi_dir, file_name)
    process_xmi_file(xmi_path)
