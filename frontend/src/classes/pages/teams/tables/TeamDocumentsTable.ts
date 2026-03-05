'use strict';

import {
    Table,
    TableRow, 
    Block, 
    Api,
    AddDocumentPopup,
    DeleteDocumentPopup,
    Button,
    Tools
} from '@src/classes';

export class TeamDocumentsTable extends Table {

    constructor(private teamId: number, parent: Block) {

        super({
            selectable: true,
            rowOptions: [
                {
                    text: 'Download',
                    event: 'download'
                },
                {
                    text: 'Delete',
                    event: 'delete'
                }
            ],
            actions: [
                {
                    text: 'Upload document',
                    event: 'add'
                }
            ],
        }, parent);

        this.on('download', (row: TableRow) => {
            this.download(row.rowData.ID);
        });

        this.on('delete', (row: TableRow) => {
            new DeleteDocumentPopup(row.rowData.ID)
                .on('done', row.delete.bind(row));
        });

        this.on('add', (button: Button) => {
            button.load();
            let popup = new AddDocumentPopup(this.teamId);
            popup.on('success', () => this.asyncPopulation());
            popup.on('hide', () => {
                button.unload();
            });
        });

        this.on('select', (row: TableRow) => {
            this.download(row.rowData.ID);
        });

        this.asyncPopulation();
    }
    
    /*
    **
    **
    */
    private async asyncPopulation() : Promise<void> {

        let rows: any[] = [];
        
        let entries: any[] = await Api.get('/team/documents', {
            teamId: this.teamId
        });

        let columns = {
            'ID': Table.NUMBER,
            'Title': Table.STRING,
            'Description': Table.STRING,
            'Uploader': Table.STRING,
            'File Name': Table.STRING
        }

        for (let entry of entries) {
            
            rows.push({
                'ID': entry.id,
                'Title': entry.title,
                'Description': entry.description,
                'Uploader': entry.uploader,
                'File Name': entry.file_name
            });
        }

        this.populate(columns, rows);
    }

    /*
    **
    **
    */
    private async download(documentId: number) : Promise<void> {

        const data = await Api.get('/document', {
            id: documentId
        });

        Tools.downloadBase64(data.file_name, data.base64, data.mime);
    }
}