#include "databasecreator.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	DatabaseCreator w;
	w.core();
	return a.exec();
}
