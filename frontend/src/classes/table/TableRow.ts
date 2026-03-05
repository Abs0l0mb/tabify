'use strict';

import { 
    Block, 
    Div,
    Table,
    TableCell,
    ContextMenu,
    ContextMenuOption 
} from '@src/classes';

export class TableRow extends Block {

    static readonly HEIGHT: number = 29;

    public cells: any = {};

    constructor(public rowData: any, private table: Table, private tbody: Block) {

        super('tr', 'data-row', tbody);

        this.onNative('click', this.select.bind(this));
        this.onNative('contextmenu', (event: MouseEvent) => {
            event.preventDefault();
            this.displayContextMenu(event);
        });

        //============
        //OPTIONS CELL
        //============

        if (this.table.configuration.rowOptions) {

            let optionsCell = new Block('td', 'options-cell', this);

            new Div('options-icon', optionsCell).onNative('click', this.displayContextMenu.bind(this));
        }

        //==========
        //DATA CELLS
        //==========

        for (let key in this.rowData) {
            
            if (this.table.configuration.hiddenColumns && this.table.configuration.hiddenColumns.includes(key))
                continue;

            this.cells[key] = new TableCell(this.rowData[key], this.table.columnsData[key], this);

            if (parseFloat(this.rowData[key]) === 0.0 
             || !this.rowData[key]
             || this.rowData[key] === 'null'
             || this.rowData[key] === '')
                this.cells[key].setData('priority', -1);
        }
    }

    /*
    **
    **
    */
    public select() : void {

        for (let row of this.table.rows)
            row.setData('selected', row.uid === this.uid ? 1 : 0);
        
        for (let row of this.table.rows)
            row.setData('options-displayed', 0);

        this.table.emit('select', this);
    }

    /*
    **
    **
    */
    public update(data: any) : void {

        for (let key in data) {

            let cell = this.cells[key];

            if (!cell)
                continue;
                
            cell.setValue(data[key]);

            this.rowData[key] = data[key];
        }

        this.emit('update-rows-data', this.rowData);
    }

    /*
    **
    **
    */
    private displayContextMenu(event: MouseEvent) : void {

        if (!this.table.configuration.rowOptions || this.table.configuration.rowOptions.length === 0)
            return;
            
        event.stopPropagation();
        
        if (!this.table.configuration.rowOptions)
            return;

        let options: ContextMenuOption[] = [];

        for (let option of this.table.configuration.rowOptions) {
            
            options.push({
                text: option.text,
                callback: () => {
                    this.table.emit(option.event, this);
                }
            });
        }

        this.setData('context-menu', 1);

        let contextMenu = new ContextMenu(event.pageX, event.pageY, options);
        
        contextMenu.on('hide', () => {
            this.setData('context-menu', 0);
        });
    }
}