#!/usr/bin/env python3

# This is a Python reimplementation of
# ctakes-misc/src/main/java/org/apache/ctakes/consumers/ExtractCuiSequences.java

import os
import pathlib
from cassis import *
from dataclasses import dataclass, astuple

xmi_dir = '/Users/Dima/Work/Data/MimicIII/Notes/Xmi/'

ident_annot_class_name = 'org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation'
umls_concept_class_name = 'org_apache_ctakes_typesystem_type_refsem_UmlsConcept'
type_system_path = 'TypeSystem.xml'

out_path = 'cuis.csv'

@dataclass
class UmlsConcept:
  """Info we need for each CUI"""

  file_name: str
  cui: str
  text: str
  coding_scheme: str
  pref_text: str
  start_offset: str
  end_offset: str

def get_ontology_concept_codes(identified_annot):
  """Extract CUIs from an identified annotation"""

  # same CUI often added multiple times
  # but must include it only once
  # key: cui, value = (coding scheme, pref text)
  codes = {}

  ontology_concept_arr = identified_annot['ontologyConceptArr']
  if not ontology_concept_arr:
    # not a umls entity, e.g. fraction annotation
    return codes

  # signs/symptoms, disease/disorders etc. have CUIs
  for ontology_concept in ontology_concept_arr.elements:
    if type(ontology_concept).__name__ == umls_concept_class_name:
      coding_scheme = ontology_concept['codingScheme']
      pref_text = ontology_concept['preferredText']
      cui = ontology_concept['cui']
      codes[cui] = (coding_scheme, pref_text)
    else:
      print('This never happens anymore, but I think it used to')

  return codes

def process_xmi_file(xmi_path, type_system):
  """This is a Python staple"""

  xmi_file = open(xmi_path, 'rb')
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
  sys_view = cas.get_view('_InitialView')
  source_file = pathlib.Path(xmi_path).stem

  umls_concepts = []
  for ident_annot in sys_view.select(ident_annot_class_name):
    text = ident_annot.get_covered_text()
    start_offset = ident_annot['begin']
    end_offset = ident_annot['end']

    codes = get_ontology_concept_codes(ident_annot)
    for cui, (coding_scheme, pref_text) in codes.items():
      umls_concept = UmlsConcept(
        file_name=source_file,
        cui=cui,
        text=text,
        coding_scheme=coding_scheme.lower(),
        pref_text=str(pref_text), # None in some cases
        start_offset=str(start_offset),
        end_offset=str(end_offset))
      umls_concepts.append(umls_concept)

  return umls_concepts

def main():
  """Main driver"""

  # load type system once for all files
  type_system_file = open(type_system_path, 'rb')
  type_system = load_typesystem(type_system_file)

  out_file = open(out_path, 'w')
  for file_name in os.listdir(xmi_dir):
    xmi_path = os.path.join(xmi_dir, file_name)
    umls_concepts = process_xmi_file(xmi_path, type_system)

    for uc in umls_concepts:
      uc_as_str = ','.join(astuple(uc)) + '\n'
      out_file.write(uc_as_str)

if __name__ == "__main__":

  main()