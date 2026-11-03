"""Render the corrected four-RQ methods/results from editable content JSON."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment.common import read_json
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
def math_run(text):
 r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r
def sub(base,index):
 node=OxmlElement('m:sSub');e=OxmlElement('m:e');e.append(math_run(base));s=OxmlElement('m:sub');s.append(math_run(index));node.extend([e,s]);return node
def summation(lower,upper,terms):
 node=OxmlElement('m:nary');pr=OxmlElement('m:naryPr');char=OxmlElement('m:chr');char.set(qn('m:val'),'∑');pr.append(char)
 if not upper:hide=OxmlElement('m:supHide');hide.set(qn('m:val'),'1');pr.append(hide)
 node.append(pr)
 for tag,text in [('sub',lower),('sup',upper)]:
  el=OxmlElement('m:'+tag);el.append(math_run(text));node.append(el)
 e=OxmlElement('m:e')
 for term in terms:e.append(term)
 node.append(e);return node
def equation(identifier):
 eq=OxmlElement('m:oMath')
 if identifier=='P036':terms=[math_run('AP = '),summation('m=1','M',[sub('P','m'),math_run(' ('),sub('R','m'),math_run(' − '),sub('R','m−1'),math_run(')')])]
 elif identifier in ['P042','P045']:
  suffix,a,b=('history','R1','R0') if identifier=='P042' else ('profile','R4','R1')
  terms=[sub('ΔAP',suffix),math_run(f' = AP({a}) − AP({b})')]
 elif identifier=='P059':
  frac=OxmlElement('m:f');num=OxmlElement('m:num');num.append(math_run('1'));den=OxmlElement('m:den');den.append(math_run('4'));frac.extend([num,den])
  terms=[sub('ΔAP','j'),math_run(' = '),frac,summation('c∈Cⱼ','',[math_run('[AP(c,j=1) − AP(c,j=0)]')])]
 else:terms=[sub('ΔAP','target'),math_run(' = '),sub('AP','target'),math_run('(R4) − '),sub('AP','target'),math_run('(R1)')]
 for term in terms:eq.append(term)
 return eq
def main():
 assert read_json(ROOT/'results/result_contract.json')['status']=='PASS'
 content=read_json(ROOT/'paper/manuscript_content.json');doc=Document();sec=doc.sections[0]
 sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=Cm(1.7);sec.bottom_margin=Cm(1.7);sec.left_margin=Cm(2);sec.right_margin=Cm(2)
 for name in ['Normal','Title','Heading 1','Heading 2']:
  st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
 doc.styles['Normal'].font.size=Pt(11);doc.styles['Normal'].paragraph_format.space_after=Pt(6);doc.styles['Normal'].paragraph_format.line_spacing=1.05
 doc.styles['Title'].font.size=Pt(17);doc.styles['Heading 1'].font.size=Pt(14);doc.styles['Heading 2'].font.size=Pt(12)
 doc.add_heading('PBDEWD зерттеуінің әдістері мен нәтижелері',0)
 doc.add_paragraph('Рецензенттік нұсқа  2026 жылғы 2 қараша\nТөрт зерттеу сұрағы бойынша түзетілген 3 және 4 бөлімдер. Барлық нәтижелер ретроспективті бағалау шегінде түсіндіріледі.')
 footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
 footer.add_run('PBDEWD  |  ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
 for block in content['blocks']:
  if 'equation' in block:
   p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
   p._p.append(equation(block['id']));p.add_run('    ('+{'P036':'1','P042':'2','P045':'3','P059':'4','P077':'5'}[block['id']]+')')
  elif 'rows' in block:
   rows=block['rows'];table=doc.add_table(rows=1,cols=len(rows[0]));table.autofit=False
   widths={3:[5.6,6.0,5.4],4:[4.7,4.1,4.1,4.1],5:[5.5,1.5,3.3,3.3,3.4],6:[4.0,1.5,2.2,2.2,2.3,4.8]}.get(len(rows[0]))
   if block['id']=='T01':widths=[4.5,2.4,2.2,2.5,2.5,2.9]
   if block['id']=='T06':widths=[7.0,1.5,2.8,2.8,2.9]
   if widths:
    for column,width in zip(table.columns,widths):column.width=Cm(width)
   last=''
   for i,values in enumerate(rows):
    cells=table.rows[0].cells if i==0 else table.add_row().cells
    values=list(values)
    if i:
     if not values[0]:values[0]=last
     else:last=values[0]
    trPr=cells[0]._tc.getparent().get_or_add_trPr();ns=OxmlElement('w:cantSplit');trPr.append(ns)
    if i==0:repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
    for j,(cell,text) in enumerate(zip(cells,values)):
     if widths:cell.width=Cm(widths[j])
     cell.text=text
     for p in cell.paragraphs:
      p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4);p.paragraph_format.keep_with_next=i==0
      for run in p.runs:run.font.size=Pt(9);run.bold=i==0
     if i==0:sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EAEAEA');cell._tc.get_or_add_tcPr().append(sh)
   doc.add_paragraph().paragraph_format.space_after=Pt(0)
  else:
   text=block['text']
   if re.match(r'^[34]\. ',text):
    p=doc.add_heading(text,1)
    if text.startswith('4. '):p.paragraph_format.page_break_before=True
   elif re.match(r'^[34]\.\d+\.',text):doc.add_heading(text,2)
   else:
    p=doc.add_paragraph(text)
    if re.match(r'^\d+-кесте',text):
     p.paragraph_format.keep_with_next=True
     for run in p.runs:run.bold=True
    if text.startswith('RQ'):
     for run in p.runs:run.bold=True
 for el in [doc._element,doc.styles.element]:
  for border in el.xpath('.//w:pBdr'):border.getparent().remove(border)
 doc.core_properties.title='PBDEWD төрт зерттеу сұрағы бойынша әдістер мен нәтижелер';doc.core_properties.author='PBDEWD зерттеу тобы'
 path=ROOT/'paper/PBDEWD_review_ready.docx';doc.save(path);print(path)
if __name__=='__main__':main()
