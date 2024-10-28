from langchain.tools.retriever import create_retriever_tool

def create_retriever_tool_from_vectorstore(vectorstore):
    retriever = vectorstore.as_retriever()
    return create_retriever_tool(
        retriever,
        "retrieve_company_docs",
        "Search and return information about AutoCare",
    )

from typing import Dict, Literal, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

# Definición de opciones para la especialidad de los médicos
EspecialidadMedica = Literal["Cardiología", "Pediatría", "Dermatología", "Oftalmología", "Ginecología"]

# Entrada para el sistema de recomendación de médicos
class MedicoInput(BaseModel):
    especialidad: EspecialidadMedica = Field(..., description="La especialidad del médico")
    ubicacion: str = Field(..., description="Ubicación geográfica del afiliado")

# Salida que muestra los detalles de los médicos recomendados
class MedicoOutput(BaseModel):
    nombre: str
    especialidad: EspecialidadMedica
    ubicacion: str
    turnos_disponibles: List[str]

# Herramienta para obtener información de médicos según especialidad y ubicación
class GetMedicoTool(BaseTool):
    name: str = "get_medico_recomendado"
    description: str = "Recomienda médicos basados en su especialidad y ubicación"
    args_schema: type[MedicoInput] = MedicoInput

    def _run(self, especialidad: EspecialidadMedica, ubicacion: str) -> Dict:
        # Datos de médicos ficticios
        medicos_data = [
            {"nombre": "Dr. Ana Gómez", "especialidad": "Cardiología", "ubicacion": "Buenos Aires", "turnos_disponibles": ["Martes 10:00", "Jueves 14:00"]},
            {"nombre": "Dr. Juan Pérez", "especialidad": "Pediatría", "ubicacion": "Rosario", "turnos_disponibles": ["Lunes 09:00", "Miércoles 11:00"]},
            {"nombre": "Dra. Laura Sánchez", "especialidad": "Dermatología", "ubicacion": "Córdoba", "turnos_disponibles": ["Viernes 12:00", "Sábado 09:00"]},
            {"nombre": "Dr. Carlos Ruiz", "especialidad": "Oftalmología", "ubicacion": "Buenos Aires", "turnos_disponibles": ["Martes 08:00", "Jueves 15:00"]},
            {"nombre": "Dra. Marta Fernández", "especialidad": "Ginecología", "ubicacion": "Mendoza", "turnos_disponibles": ["Lunes 14:00", "Viernes 10:00"]}
        ]

        # Filtrar médicos según especialidad y ubicación
        medicos_recomendados = [
            medico for medico in medicos_data 
            if medico["especialidad"] == especialidad and medico["ubicacion"] == ubicacion
        ]

        if not medicos_recomendados:
            raise ValueError(f"No hay médicos disponibles con especialidad {especialidad} en {ubicacion}.")

        return [MedicoOutput(**medico) for medico in medicos_recomendados]

# Función para crear la herramienta de recomendación de médicos
def create_get_medico_tool():
    return GetMedicoTool()