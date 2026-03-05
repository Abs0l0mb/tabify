'use strict';

import {
    Table,
    TableRow, 
    Block, 
    Api,
    AddTeamPopup,
    EditTeamPopup,
    DeleteTeamPopup,
    Button,
    AddDocumentForAllTeamsPopup
} from '@src/classes';

export class TeamsTable extends Table {

    constructor(parent: Block) {

        super({
            selectable: true,
            rowOptions: [
                {
                    text: 'Edit',
                    event: 'edit'
                },
                {
                    text: 'Delete',
                    event: 'delete'
                }
            ],
            actions: [
                {
                    text: 'Add team',
                    event: 'add'
                },
                {
                    text: 'Upload document for all teams',
                    event: 'upload-document-for-all-teams'
                }
            ],
        }, parent);

        this.on('edit', (row: TableRow) => {
            new EditTeamPopup(row.rowData.ID)
                .on('success', () => this.asyncPopulation());
        });

        this.on('delete', (row: TableRow) => {
            new DeleteTeamPopup(row.rowData.ID)
                .on('done', row.delete.bind(row));
        });

        this.on('add', (button: Button) => {
            button.load();
            let popup = new AddTeamPopup();
            popup.on('success', () => this.asyncPopulation());
            popup.on('hide', () => {
                button.unload();
            });
        });

        this.on('upload-document-for-all-teams', (button: Button) => {
            button.load();
            let popup = new AddDocumentForAllTeamsPopup();
            popup.on('hide', () => {
                button.unload();
            });
        });

        this.asyncPopulation();
    }
    
    /*
    **
    **
    */
    private async asyncPopulation() : Promise<void> {

        let rows: any[] = [];
        
        let entries: any[] = await Api.get('/teams');

        let columns = {
            'ID': Table.NUMBER,
            'Name': Table.STRING
        }

        for (let entry of entries) {
            
            rows.push({
                'ID': entry.id,
                'Name': entry.name
            });
        }

        this.populate(columns, rows);
    }
}