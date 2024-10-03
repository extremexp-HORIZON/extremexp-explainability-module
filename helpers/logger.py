import os.path
import logging
import logging.config
from helpers.config import ConfigHelper

class LoggerHelper:
    
    def init_logger():
        """ Initiates logging
        """

        # Crear ruta de logs
        try:
            path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../")
            if not os.path.isdir(path + "/logs"):
                os.mkdir(path + "/logs")
                print("Successfully created the directory %s/logs" % path)
        except OSError:
            print("Creation of the directory %s/logs failed" % path)
      

        # Incializar el logger
        try:
            logging_configuration = ConfigHelper.instance("logging")
            logging.config.dictConfig(logging_configuration)
        except (ValueError, TypeError, AttributeError, ImportError) as exc:
            print("Error al instanciar o crear la configuración de logging %s", str(exc))
