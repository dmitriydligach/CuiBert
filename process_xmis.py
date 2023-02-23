#!/usr/bin/env python3

import os
from cassis import *

# xmi_path = '/Users/Dima/Work/Data/MimicIII/Notes/XmiSample/1000.txt.xmi'
xmi_dir = '/Users/Dima/Work/Data/MimicIII/Notes/Xmi/'
identified_annot_class = 'org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation'
umls_concept_class = 'org_apache_ctakes_typesystem_type_refsem_UmlsConcept'
type_system_path = 'TypeSystem.xml'

def get_ontology_concept_codes(identified_annot):
  """Extract CUIs from an identified annotation"""

  codes = set()

  ontology_concept_arr = identified_annot['ontologyConceptArr']
  if not ontology_concept_arr:
    # not a umls entity, e.g. fraction annotation
    return codes

  for ontology_concept in ontology_concept_arr.elements:
    if type(ontology_concept).__name__ == umls_concept_class:
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
  for ident_annot in sys_view.select(identified_annot_class):
    text = ident_annot.get_covered_text().replace('\n', '')
    for code in get_ontology_concept_codes(ident_annot):
      print(text, '/', code)
      cuis.append(code)

if __name__ == "__main__":

  for file_name in os.listdir(xmi_dir):
    xmi_path = os.path.join(xmi_dir, file_name)
    process_xmi_file(xmi_path)
